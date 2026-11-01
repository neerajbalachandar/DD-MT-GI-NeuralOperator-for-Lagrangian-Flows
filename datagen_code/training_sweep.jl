#!/usr/bin/env julia

using Printf
using TOML
using SHA
using LinearAlgebra: I
import FLOWVPM
import GeometricTools

const ROOT = @__DIR__
const vpm = FLOWVPM
const gt = GeometricTools

module VortexRingCode
include(joinpath(@__DIR__, "vortexrings", "vortexrings.jl"))
end

module RoundJetCode
include(joinpath(@__DIR__, "roundjet", "roundjet.jl"))
end

module UtilitiesCode
include(joinpath(@__DIR__, "utilities", "utilities.jl"))
end

const DATASET = length(ARGS) > 0 ? ARGS[1] : get(ENV, "DATASET", "all")
const OUTPUT_ROOT = abspath(get(ENV, "OUTPUT_ROOT", joinpath(ROOT, "training_data")))
const GRID_POINTS = parse(Int, get(ENV, "GRID_POINTS", "32"))
const SAVE_EVERY = parse(Int, get(ENV, "SAVE_EVERY", "4"))
const CASE_LIMIT = parse(Int, get(ENV, "CASE_LIMIT", "0"))
const VORTEX_STEPS = parse(Int, get(ENV, "VORTEX_STEPS", "1000"))
const VORTEX_RTOT = parse(Float64, get(ENV, "VORTEX_RTOT", "10"))
const ROUNDJET_STEPS_PER_D = parse(Int, get(ENV, "ROUNDJET_STEPS_PER_D", "100"))
const ROUNDJET_TRAVEL_D = parse(Float64, get(ENV, "ROUNDJET_TRAVEL_D", "10"))
const ROUNDJET_RE = parse(Float64, get(ENV, "ROUNDJET_RE", "10000"))
const ROUNDJET_RBF_ITMAX = parse(Int, get(ENV, "ROUNDJET_RBF_ITMAX", "200"))
const NGRID = GRID_POINTS^3
const NDIVS = fill(GRID_POINTS - 1, 3) # GeometricTools counts cells; this produces GRID_POINTS nodes per axis.

DATASET in ("all", "vortexrings", "roundjet") || error("Dataset must be all, vortexrings, or roundjet")
GRID_POINTS >= 2 || error("GRID_POINTS must be at least 2")
SAVE_EVERY >= 1 || error("SAVE_EVERY must be at least 1")
VORTEX_STEPS % SAVE_EVERY == 0 || error("VORTEX_STEPS must be divisible by SAVE_EVERY so particle and field snapshots align")
ceil(Int, ROUNDJET_TRAVEL_D * ROUNDJET_STEPS_PER_D) % SAVE_EVERY == 0 ||
    error("Roundjet step count must be divisible by SAVE_EVERY so particle and field snapshots align")

function safe_tag(x)
    return replace(@sprintf("%.3f", x), "." => "p", "-" => "m")
end

function new_case_dir(simulation, case_id, metadata)
    # Hash each complete initial condition so all snapshots from one trajectory
    # remain together. The 73/15/12 buckets compensate for the perturbation
    # family being reserved for test, keeping the overall split near 70/15/15.
    config = join((string(k, "=", metadata[k]) for k in sort!(collect(keys(metadata)))), "|")
    bucket = mod(parse(UInt, bytes2hex(sha256(config))[1:8]; base=16), UInt(100))
    # Image A_2 reserves perturbed geometries for testing generalization.
    split = get(metadata, "family", "") == "three_dimensional_perturbation" ?
        "test" : bucket < 73 ? "train" : bucket < 88 ? "validation" : "test"
    case_dir = joinpath(OUTPUT_ROOT, split, simulation, case_id)
    ispath(case_dir) && error("Output already exists: $case_dir (move it or choose another OUTPUT_ROOT)")
    mkpath(dirname(case_dir))
    metadata["split"] = split
    metadata["split_rule"] = "Complete-config SHA-256 buckets; perturbation family held out for test"
    metadata["simulation"] = simulation
    metadata["case_id"] = case_id
    metadata["grid_points"] = [GRID_POINTS, GRID_POINTS, GRID_POINTS]
    metadata["save_every_steps"] = SAVE_EVERY
    metadata["fidelity_profile"] = "mid_high"
    metadata["field_components"] = ["U", "W"]
    return case_dir
end

function record_case_metadata!(case_dir, metadata)
    open(joinpath(case_dir, "case.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function reconstruct_fields!(pfield, case_dir, run_name, pmin, pmax)
    grid = gt.Grid(pmin, pmax, NDIVS)
    nsteps = pfield.nt
    nums = collect(0:SAVE_EVERY:nsteps)
    println("Reconstructing U and W on a fixed $(GRID_POINTS)^3 grid for $(length(nums)) snapshots")
    UtilitiesCode.computefluiddomain(pfield, nums, case_dir, run_name, [grid];
        save_path=case_dir,
        file_pref="fields",
        grid_names=["_fdom"],
        f_sigma=0.5,
        scale_sigma=1.0,
        verbose=true)
end

function run_vortex_case(case_id, metadata, nrings, circulations, radii, Ntotal, spacing;
                         sigmas=0.10 .* radii, nsteps=VORTEX_STEPS,
                         Rtot=VORTEX_RTOT, lateral=false)
    case_dir = new_case_dir("vortexrings", case_id, metadata)
    per_ring = Ntotal ÷ nrings
    Ntotal % nrings == 0 || error("Total N must divide evenly over the rings")

    ARs = ones(nrings)
    Rcrosss = 0.10 .* radii
    Nphis = fill(per_ring, nrings)
    ncs = ones(Int, nrings)
    extra_ncs = zeros(Int, nrings)
    Os = lateral ? [[spacing * (ri - 1), 0.0, 0.0] for ri in 1:nrings] :
                   [[0.0, 0.0, spacing * (ri - 1)] for ri in 1:nrings]
    Oaxiss = [Matrix{Float64}(I, 3, 3) for _ in 1:nrings]
    fmm = vpm.FMM(; p=6, ncrit=60, theta=0.3, shrink_recenter=true)
    solver = (
        formulation=vpm.cVPM,
        SFS=vpm.noSFS,
        relaxation=vpm.correctedpedrizzetti,
        kernel=vpm.winckelmans,
        viscous=vpm.Inviscid(),
        transposed=true,
        integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm,
        fmm=fmm,
    )
    # The particle field is later augmented by all grid nodes during
    # `computefluiddomain`, so reserve capacity for the initialized ring
    # particles plus the temporary reconstruction particles. Using only
    # `Ntotal + NGRID` underestimates the true count because the actual ring
    # discretization is larger than the nominal `Ntotal` parameter.
    initial_particles = sum(
        ri -> VortexRingCode.number_particles(Nphis[ri], ncs[ri]; extra_nc=extra_ncs[ri]),
        1:nrings,
    )
    maxparticles = initial_particles + NGRID + 4096

    println("\n[$case_id] N=$Ntotal, spacing/R=$spacing, circulations=$circulations")
    pfield = VortexRingCode.run_vortexring_simulation(nrings, circulations, radii, ARs, Rcrosss,
        Nphis, ncs, extra_ncs, sigmas, Os, Oaxiss;
        Re=nothing,
        nref=1,
        nsteps=nsteps,
        Rtot=Rtot,
        beta=0.5,
        faux=1.0,
        save_path=case_dir,
        run_name="pfield",
        prompt=false,
        calc_monitors=false,
        verbose=true,
        verbose_nsteps=100,
        nsteps_save=SAVE_EVERY,
        maxparticles=maxparticles,
        pfieldargs=solver)

    metadata["final_particle_count"] = vpm.get_np(pfield)
    metadata["completed_steps"] = pfield.nt
    record_case_metadata!(case_dir, metadata)
    pmin, pmax = vortex_field_bounds(Os, radii, circulations, Rtot)
    metadata["field_bounds_min"] = pmin
    metadata["field_bounds_max"] = pmax
    record_case_metadata!(case_dir, metadata)
    reconstruct_fields!(pfield, case_dir, "pfield", pmin, pmax)
end

function vortex_field_bounds(origins, radii, circulations, travel_radii)
    rmax = maximum(radii)
    margin = 2.5 * rmax
    xcenters = first.(origins)
    ycenters = getindex.(origins, 2)
    zcenters = getindex.(origins, 3)
    ztravel = [z + sign(gamma) * travel_radii * rmax
               for (z, gamma) in zip(zcenters, circulations)]
    pmin = [minimum(xcenters) - margin, minimum(ycenters) - margin,
            min(minimum(zcenters), minimum(ztravel)) - margin]
    pmax = [maximum(xcenters) + margin, maximum(ycenters) + margin,
            max(maximum(zcenters), maximum(ztravel)) + margin]
    return pmin, pmax
end

function run_vortexrings_sweep()
    case_index = 0
    case_limit = CASE_LIMIT > 0 ? CASE_LIMIT : typemax(Int)

    # Mid/high particle counts expand the image-defined leapfrogging sweep.
    for N in (160, 320, 640), dR in (0.75, 1.0, 1.25, 1.5, 2.0), ratio in (1.0, 0.75, 0.5)
        case_index += 1
        if case_index > case_limit
            return
        end
        case_id = "leapfrog_N$(N)_d$(safe_tag(dR))_g$(safe_tag(ratio))"
        run_vortex_case(case_id,
            Dict("family" => "leapfrogging", "total_particles" => N,
                 "radius" => 1.0, "spacing_over_radius" => dR,
                 "circulation_ratio" => [1.0, ratio]),
            2, [10.0, 10.0 * ratio], [1.0, 1.0], N, dR)
    end

    # Image-defined isolated-ring sweep over particle count, circulation, and core size.
    for N in (160, 320, 640), gamma in (5.0, 10.0, 20.0), sigmaR in (0.02, 0.04, 0.08)
        case_index += 1
        case_index > case_limit && return
        case_id = "isolated_N$(N)_g$(safe_tag(gamma))_sigma$(safe_tag(sigmaR))"
        run_vortex_case(case_id,
            Dict("family" => "isolated_ring", "total_particles" => N,
                 "radius" => 1.0, "circulation" => gamma, "sigma_over_radius" => sigmaR),
            1, [gamma], [1.0], N, 0.0; sigmas=[sigmaR])
    end

    # Signed/unequal rings are repeated at multiple particle resolutions.
    for N in (160, 320), dR in (0.75, 1.0, 1.5, 2.0), ratio in (-1.0, -0.75, -0.5)
        case_index += 1
        case_index > case_limit && return
        case_id = "signed_N$(N)_d$(safe_tag(dR))_g$(safe_tag(ratio))"
        run_vortex_case(case_id,
            Dict("family" => "signed_two_rings", "total_particles" => N,
                 "radius" => 1.0, "spacing_over_radius" => dR,
                 "circulation_ratio" => [1.0, ratio]),
            2, [10.0, 10.0 * ratio], [1.0, 1.0], N, dR)
    end

    # Toroidal-vorticity ring using FLOWVPM's Cartesian-grid initializer.
    # Preserve the example's Gaussian core, Rcross/R=0.1, sigma=Rcross,
    # and 5% vorticity cutoff while sweeping circulation, radius, and dx.
    for radius in (0.75, 1.0, 1.25), gamma in (5.0, 10.0, 20.0), dxratio in (0.5, 0.75, 1.0)
        case_index += 1
        case_index > case_limit && return
        run_toroidal_case(radius, gamma, dxratio)
    end

    # Freely evolving pair in unbounded space, with controlled center spacing,
    # circulation ratio, and particle count. The pair is laterally offset.
    for N in (160, 320, 640), separationR in (2.5, 3.0, 4.0), ratio in (-1.0, -0.75, -0.5)
        case_index += 1
        case_index > case_limit && return
        case_id = "free_pair_N$(N)_sep$(safe_tag(separationR))_g$(safe_tag(ratio))"
        metadata = Dict("family" => "free_vortex_pair", "total_particles" => N,
            "radius" => 1.0, "center_separation_over_radius" => separationR,
            "circulation_ratio" => [1.0, ratio], "core_radius_over_radius" => 0.10)
        run_vortex_case(case_id, metadata, 2, [10.0, 10.0 * ratio],
            [1.0, 1.0], N, separationR; lateral=true)
    end

    # Advanced 3D deformation cases are held out in test as recommended by A_2.
    for N in (160, 320, 640), amplitudeR in (0.02, 0.05, 0.10), mode in (2, 3)
        case_index += 1
        case_index > case_limit && return
        run_perturbed_ring_case(N, amplitudeR, mode)
    end
end

function run_toroidal_case(radius, gamma, dxratio)
    case_id = "toroidal_R$(safe_tag(radius))_g$(safe_tag(gamma))_dx$(safe_tag(dxratio))"
    metadata = Dict("family" => "toroidal_vorticity", "radius" => radius,
        "circulation" => gamma, "aspect_ratio" => 1.0,
        "core_radius_over_radius" => 0.10, "grid_spacing_over_core_radius" => dxratio,
        "smoothing_radius_over_core_radius" => 1.0, "minWfraction" => 0.05)
    case_dir = new_case_dir("vortexrings", case_id, metadata)
    fmm = vpm.FMM(; p=6, ncrit=60, theta=0.3, shrink_recenter=true)
    solver = (formulation=vpm.cVPM, SFS=vpm.noSFS,
        relaxation=vpm.correctedpedrizzetti, kernel=vpm.winckelmans,
        viscous=vpm.Inviscid(), transposed=true, integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm, fmm=fmm)
    Rcross = 0.10 * radius

    println("\n[$case_id] toroidal grid initialization")
    pfield = VortexRingCode.run_vortexring_grid_simulation(1, [gamma], [radius], [1.0],
        [Rcross], [dxratio], [Rcross], 0.05, [[0.0, 0.0, 0.0]],
        [Matrix{Float64}(I, 3, 3)];
        nsteps=VORTEX_STEPS, Rtot=VORTEX_RTOT, beta=0.5, faux=1.0,
        maxparticles=500_000 + NGRID, pfieldargs=solver,
        save_path=case_dir, run_name="pfield", prompt=false,
        calc_monitors=false, verbose=true, verbose_nsteps=100,
        nsteps_save=SAVE_EVERY)

    pmin, pmax = vortex_field_bounds([[0.0, 0.0, 0.0]], [radius], [gamma], VORTEX_RTOT)
    metadata["field_bounds_min"] = pmin
    metadata["field_bounds_max"] = pmax
    record_case_metadata!(case_dir, metadata)
    reconstruct_fields!(pfield, case_dir, "pfield", pmin, pmax)
end

function run_perturbed_ring_case(N, amplitudeR, mode)
    case_id = "perturbed_N$(N)_amp$(safe_tag(amplitudeR))_mode$(mode)"
    metadata = Dict("family" => "three_dimensional_perturbation", "total_particles" => N,
        "radius" => 1.0, "circulation" => 10.0, "core_radius_over_radius" => 0.10,
        "perturbation_amplitude_over_radius" => amplitudeR,
        "azimuthal_mode" => mode)
    case_dir = new_case_dir("vortexrings", case_id, metadata)
    fmm = vpm.FMM(; p=6, ncrit=60, theta=0.3, shrink_recenter=true)
    solver = (formulation=vpm.cVPM, SFS=vpm.noSFS,
        relaxation=vpm.correctedpedrizzetti, kernel=vpm.winckelmans,
        viscous=vpm.Inviscid(), transposed=true, integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm, fmm=fmm)
    pfield = vpm.ParticleField(N + NGRID; solver...)
    R = 1.0
    Rcross = 0.10
    Nphi = N
    VortexRingCode.addvortexring(pfield, 10.0, R, 1.0, Rcross, Nphi, 1, Rcross;
        O=zeros(3), Oaxis=Matrix{Float64}(I, 3, 3), verbose=true)

    # Apply a small out-of-plane sinusoidal deformation to the centerline and
    # rotate each particle's circulation vector to follow the deformed tangent.
    for particle in vpm.iterate(pfield)
        X = vpm.get_X(particle)
        Gamma = vpm.get_Gamma(particle)
        x, y = X[1], X[2]
        phi = atan(y, x)
        radius_here = max(hypot(x, y), eps())
        dz_dphi = -amplitudeR * mode * R * sin(mode * phi)
        X[3] += amplitudeR * R * cos(mode * phi)
        gamma_xy = hypot(Gamma[1], Gamma[2])
        Gamma[3] = gamma_xy * dz_dphi / radius_here
    end

    Uref = VortexRingCode.Uring(10.0, R, Rcross, 0.5)
    nsteps, Rtot = VORTEX_STEPS, VORTEX_RTOT
    dt = (Rtot / Uref) / nsteps
    println("\n[$case_id] held-out three-dimensional ring perturbation")
    VortexRingCode.run_vortexring_simulation(pfield, nsteps, dt, 1,
        [Nphi], [1], [0];
        save_path=case_dir, run_name="pfield", prompt=false,
        calc_monitors=false, verbose=true, verbose_nsteps=100,
        nsteps_save=SAVE_EVERY)

    pmin, pmax = vortex_field_bounds([[0.0, 0.0, 0.0]], [R], [10.0], Rtot)
    metadata["field_bounds_min"] = pmin
    metadata["field_bounds_max"] = pmax
    record_case_metadata!(case_dir, metadata)
    reconstruct_fields!(pfield, case_dir, "pfield", pmin, pmax)
end

function run_roundjet_case(case_id, metadata, diameter, exit_speed, dxotheta, overlap)
    case_dir = new_case_dir("roundjet", case_id, metadata)
    fmm = vpm.FMM(; p=6, ncrit=60, theta=0.3, shrink_recenter=true)
    maxparticles = roundjet_particle_capacity(diameter, exit_speed, dxotheta, overlap)
    viscous = vpm.CoreSpreading(-1.0, -1.0, RoundJetCode.zeta_fmm_compat;
        itmax=20, tol=1e-2, iterror=false)

    thetaod = 0.025 # Official example baseline; this driver has no inlet-forcing input.
    println("\n[$case_id] D=$diameter, Uexit=$exit_speed, dx/theta=$dxotheta, overlap=$overlap")
    metadata["Reynolds_number"] = ROUNDJET_RE
    metadata["steps_per_diameter"] = ROUNDJET_STEPS_PER_D
    metadata["travel_diameters"] = ROUNDJET_TRAVEL_D
    pmin = [-2.5 * diameter, -2.5 * diameter, -2.0 * diameter]
    pmax = [2.5 * diameter, 2.5 * diameter, (ROUNDJET_TRAVEL_D + 2.0) * diameter]
    metadata["field_bounds_min"] = pmin
    metadata["field_bounds_max"] = pmax
    pfield = RoundJetCode.run_roundjet_simulation(diameter, exit_speed, 0.0;
        thetaod=thetaod,
        steps_per_d=ROUNDJET_STEPS_PER_D,
        d_travel_tot=ROUNDJET_TRAVEL_D,
        dxotheta=dxotheta,
        overlap=overlap,
        minWfraction=1e-2,
        maxRoR=1.0,
        rbf=true,
        rbf_optargs=[(:itmax, ROUNDJET_RBF_ITMAX), (:tol, 2.5e-2),
                     (:iterror, true), (:verbose, true), (:debug, false)],
        Re=ROUNDJET_RE,
        maxparticles=maxparticles,
        pfieldargs=(fmm=fmm, kernel=vpm.gaussianerf, viscous=viscous),
        save_path=case_dir,
        run_name="pfield",
        prompt=false,
        verbose=true,
        verbose_nsteps=100,
        nsteps_save=SAVE_EVERY)

    metadata["final_particle_count"] = vpm.get_np(pfield)
    metadata["completed_steps"] = pfield.nt
    record_case_metadata!(case_dir, metadata)
    reconstruct_fields!(pfield, case_dir, "pfield", pmin, pmax)
end

function roundjet_particle_capacity(diameter, exit_speed, dxotheta, overlap)
    # Size for initialization and field reconstruction. The round-jet driver
    # grows its particle buffer as new inlet particles arrive over the trajectory.
    thetaod = 0.025
    maxRoR = 1.0
    max_zsigma = 12.0
    minWfraction = 1e-2
    steps_per_d = ROUNDJET_STEPS_PER_D
    d = diameter
    theta = thetaod * d
    dx = dxotheta * thetaod * d
    sigma = overlap * dx
    R = d / 2
    maxR = maxRoR * R
    Nphi = ceil(Int, 2pi * R / dx)
    NR = ceil(Int, maxR / dx)
    dr = maxR / NR
    dt = d / steps_per_d / exit_speed
    Wpeak_bound = exit_speed / theta

    initial_particles = 0
    # Match the round-jet initializer's annulus threshold, with a factor-of-two
    # margin on the threshold to cover its sampled Wpeak and quadrature rounding.
    for ri in 1:NR
        rlo, rup = dr * (ri - 1), dr * ri
        rc = (rlo + rup) / 2
        wmean = exit_speed * (tanh((rup - R) / theta) - tanh((rlo - R) / theta)) / (rup - rlo)
        abs(wmean) / Wpeak_bound >= minWfraction / 2 || continue

        centerline_speed = exit_speed * tanh((R - rc) / theta)
        dz = centerline_speed * dt
        # Match the initializer's maximum streamwise extent and keep one extra layer.
        nz = ceil(Int, max_zsigma * sigma / dz)
        initial_particles += Nphi * (nz + 1)
    end

    # Include the initial static wake, field-grid nodes, and headroom for
    # threshold/rounding differences. Dynamic inlet particles expand on demand.
    return ceil(Int, 1.25 * initial_particles) + NGRID + 4096
end

function run_roundjet_sweep()
    # Retain the official theta/D=0.025 and 10D duration. Sweep supported
    # operating conditions, particle resolution, and core overlap. The current
    # driver exposes no inlet perturbation/forcing control.
    case_index = 0
    case_limit = CASE_LIMIT > 0 ? CASE_LIMIT : typemax(Int)
    for diameter in (1.0, 2.0), exit_speed in (0.75, 1.0, 1.25), dxotheta in (1/6, 1/8), overlap in (2.0, 2.4, 3.0)
        case_index += 1
        if case_index > case_limit
            return
        end
        case_id = "jet_D$(safe_tag(diameter))_U$(safe_tag(exit_speed))_dx$(safe_tag(dxotheta))_ov$(safe_tag(overlap))"
        run_roundjet_case(case_id,
            Dict("family" => "roundjet", "diameter" => diameter,
                 "exit_speed" => exit_speed, "coflow_speed" => 0.0,
                 "momentum_thickness_over_diameter" => 0.025,
                 "particle_spacing_over_momentum_thickness" => dxotheta,
                 "particle_overlap" => overlap,
                 "Reynolds_number" => ROUNDJET_RE),
            diameter, exit_speed, dxotheta, overlap)
    end
end

mkpath(OUTPUT_ROOT)
println("Train/validation/test output: $OUTPUT_ROOT/{train,validation,test}")
println("Grid: $(GRID_POINTS) x $(GRID_POINTS) x $(GRID_POINTS) nodes; snapshots every $SAVE_EVERY steps")

if DATASET in ("all", "vortexrings")
    run_vortexrings_sweep()
end
if DATASET in ("all", "roundjet")
    run_roundjet_sweep()
end

println("Training-data generation complete.")
