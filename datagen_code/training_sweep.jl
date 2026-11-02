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
const VORTEX_STEPS = parse(Int, get(ENV, "VORTEX_STEPS", "720"))
const VORTEX_RTOT = parse(Float64, get(ENV, "VORTEX_RTOT", "7.2"))
const ROUNDJET_STEPS_PER_D = parse(Int, get(ENV, "ROUNDJET_STEPS_PER_D", "100"))
const ROUNDJET_TRAVEL_D = parse(Float64, get(ENV, "ROUNDJET_TRAVEL_D", "7.2"))
const ROUNDJET_RE = parse(Float64, get(ENV, "ROUNDJET_RE", "10000"))
const ROUNDJET_RBF_ITMAX = parse(Int, get(ENV, "ROUNDJET_RBF_ITMAX", "200"))
const RESUME = lowercase(get(ENV, "RESUME", "1")) in ("1", "true", "yes")
const NGRID = GRID_POINTS^3
const CASE_PLAN = Dict{String,Dict{String,Any}}()

DATASET in ("all", "vortexrings", "roundjet") || error("Dataset must be all, vortexrings, or roundjet")
GRID_POINTS >= 2 || error("GRID_POINTS must be at least 2")
SAVE_EVERY >= 1 || error("SAVE_EVERY must be at least 1")
VORTEX_STEPS % SAVE_EVERY == 0 || error("VORTEX_STEPS must be divisible by SAVE_EVERY so particle and field snapshots align")
ceil(Int, ROUNDJET_TRAVEL_D * ROUNDJET_STEPS_PER_D) % SAVE_EVERY == 0 ||
    error("Roundjet step count must be divisible by SAVE_EVERY so particle and field snapshots align")

function generation_fmm()
    # FLOWVPM's regularization-error root finder can fail when its bracket does
    # not straddle a root (seen in long runs around step 200). Use the documented
    # default rho/sigma value and disable only that fragile auto-tuning step.
    return vpm.FMM(; p=6, ncrit=60, theta=0.3, shrink_recenter=true,
                   autotune_reg_error=false, default_rho_over_sigma=1.0)
end

function safe_tag(x)
    return replace(@sprintf("%.3f", x), "." => "p", "-" => "m")
end

function new_case_dir(simulation, case_id, metadata)
    plan = get(CASE_PLAN, case_id, Dict{String,Any}())
    split = String(get(plan, "split", "train"))
    metadata["evaluation_categories"] = get(plan, "categories", [split == "test" ? "in_regime" : split])
    metadata["split_rule"] = "Existing case split preserved; all frames from a case stay in one split"
    folder_name = case_id
    if split == "test" && !isempty(get(plan, "categories", String[]))
        categories = get(plan, "categories", String[])
        prefix = "temporal_superresolution" in categories ? "test_temporal_sr__" :
            "spatial_superresolution" in categories ? "test_spatial_sr__" :
            "outside_training_regime" in categories ? "test_outside_regime__" :
            "unseen_boundary_condition" in categories ? "test_unseen_boundary__" :
            ("unseen_operating_condition" in categories || "unseen_Reynolds_number" in categories) ? "test_unseen_operation__" :
            "unseen_initial_condition" in categories ? "test_unseen_initial__" :
            "unseen_case" in categories ? "test_unseen_case__" : ""
        isempty(prefix) || (folder_name = prefix * case_id)
    end
    case_dir = joinpath(OUTPUT_ROOT, split, simulation, folder_name)
    metadata["split"] = split
    metadata["simulation"] = simulation
    metadata["case_id"] = case_id
    metadata["grid_points"] = [case_grid_points(case_id), case_grid_points(case_id), case_grid_points(case_id)]
    metadata["save_every_steps"] = case_save_every(case_id)
    metadata["planned_steps"] = Int(get(plan, "steps", VORTEX_STEPS))
    metadata["fidelity_profile"] = "mid_high"
    metadata["field_components"] = ["U", "W"]
    if ispath(case_dir)
        if RESUME && case_is_complete(case_dir, metadata)
            println("[resume] complete case already exists; skipping: $case_id")
            return nothing
        end
        println("[resume] removing incomplete case before regeneration: $case_dir")
        rm(case_dir; recursive=true, force=true)
    end
    mkpath(case_dir)
    return case_dir
end

function record_case_metadata!(case_dir, metadata)
    open(joinpath(case_dir, "case.toml"), "w") do io
        TOML.print(io, metadata)
    end
end

function reconstruct_fields!(pfield, case_dir, run_name, pmin, pmax)
    grid_points = Int(get(TOML.parsefile(joinpath(case_dir, "case.toml")), "grid_points", [GRID_POINTS])[1])
    save_every = Int(get(TOML.parsefile(joinpath(case_dir, "case.toml")), "save_every_steps", SAVE_EVERY))
    grid = gt.Grid(pmin, pmax, fill(grid_points - 1, 3))
    nsteps = pfield.nt
    nums = collect(0:save_every:nsteps)
    println("Reconstructing U and W on a fixed $(grid_points)^3 grid for $(length(nums)) snapshots")
    UtilitiesCode.computefluiddomain(pfield, nums, case_dir, run_name, [grid];
        save_path=case_dir,
        file_pref="fields",
        grid_names=["_fdom"],
        f_sigma=0.5,
        scale_sigma=1.0,
        verbose=true)
    touch(joinpath(case_dir, "case.complete"))
end

function case_save_every(case_id)
    return Int(get(get(CASE_PLAN, case_id, Dict{String,Any}()), "save_every", SAVE_EVERY))
end
function case_grid_points(case_id)
    return Int(get(get(CASE_PLAN, case_id, Dict{String,Any}()), "grid_points", GRID_POINTS))
end

function case_is_complete(case_dir, metadata)
    steps = Int(get(metadata, "completed_steps", get(metadata, "planned_steps", VORTEX_STEPS)))
    steps = Int(get(metadata, "planned_steps", steps))
    stride = Int(get(metadata, "save_every_steps", SAVE_EVERY))
    expected = collect(0:stride:steps)
    all(isfile(joinpath(case_dir, "pfield.$(step).h5")) &&
        isfile(joinpath(case_dir, "pfield.$(step).xmf")) &&
        isfile(joinpath(case_dir, "fields_fdom.$(step).h5")) &&
        isfile(joinpath(case_dir, "fields_fdom.$(step).xmf")) for step in expected)
end

function run_vortex_case(case_id, metadata, nrings, circulations, radii, Ntotal, spacing;
                         sigmas=0.10 .* radii, nsteps=VORTEX_STEPS,
                         Rtot=VORTEX_RTOT, lateral=false)
    case_dir = new_case_dir("vortexrings", case_id, metadata)
    case_dir === nothing && return
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
    fmm = generation_fmm()
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
    field_nodes = case_grid_points(case_id)^3
    maxparticles = initial_particles + field_nodes + 4096

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
        nsteps_save=case_save_every(case_id),
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

function evenly_select(ids, count; required=String[])
    count <= length(ids) || error("Cannot select $count cases from $(length(ids)) candidates")
    chosen = Set(String.(required))
    for id in chosen
        id in ids || error("Required resume case $id is not in the parameter sweep")
    end
    length(chosen) <= count || error("Required existing cases exceed family budget")
    for index in round.(Int, range(1, length(ids); length=count))
        push!(chosen, ids[index])
        length(chosen) == count && break
    end
    if length(chosen) < count
        for id in ids
            push!(chosen, id)
            length(chosen) == count && break
        end
    end
    return [id for id in ids if id in chosen]
end

function prepare_case_plan!()
    empty!(CASE_PLAN)
    function plan_case!(id, simulation, split, categories; save_every=SAVE_EVERY,
                        grid_points=GRID_POINTS, steps=VORTEX_STEPS, kwargs...)
        plan = Dict{String,Any}("simulation" => simulation, "split" => split,
            "categories" => categories, "save_every" => save_every,
            "grid_points" => grid_points, "steps" => steps)
        for (key, value) in kwargs
            plan[String(key)] = value
        end
        CASE_PLAN[id] = plan
    end

    # Twenty vortex-ring cases: seven representative existing training cases,
    # four validation cases (including one N=640 resolution series member),
    # two existing in-range tests, and seven diverse additions.
    for id in (
        "leapfrog_N160_d0p750_g0p500", "leapfrog_N160_d1p000_g0p750",
        "leapfrog_N160_d2p000_g1p000", "leapfrog_N320_d0p750_g0p750",
        "leapfrog_N320_d1p000_g0p500", "leapfrog_N320_d1p000_g1p000",
        "leapfrog_N320_d1p250_g0p750")
        plan_case!(id, "vortexrings", "train", ["in_regime"])
    end
    for id in (
        "leapfrog_N160_d0p750_g0p750", "leapfrog_N160_d2p000_g0p500",
        "leapfrog_N320_d0p750_g1p000", "leapfrog_N640_d0p750_g1p000")
        plan_case!(id, "vortexrings", "validate", ["validation_in_regime"])
    end
    for id in ("leapfrog_N160_d1p500_g1p000", "leapfrog_N320_d1p250_g1p000")
        plan_case!(id, "vortexrings", "test", ["in_regime"])
    end
    plan_case!("isolated_N320_g20p000_sigma0p080", "vortexrings", "train", ["in_regime"])
    plan_case!("signed_N320_d2p000_gm1p000", "vortexrings", "train", ["in_regime"])
    plan_case!("leapfrog_N320_d2p500_g0p250", "vortexrings", "test",
        ["outside_training_regime", "unseen_operating_condition"])
    plan_case!("perturbed_N320_amp0p100_mode3", "vortexrings", "test",
        ["unseen_case", "unseen_initial_condition", "temporal_superresolution"]; steps=180, save_every=1)
    plan_case!("leapfrog_spatial_N320_d1p000_g1p000", "vortexrings", "test",
        ["spatial_superresolution"]; grid_points=64)
    plan_case!("toroidal_R1p500_g20p000_dx0p500", "vortexrings", "test",
        ["unseen_case", "outside_training_regime"])
    plan_case!("free_pair_N320_sep5p000_gm1p000", "vortexrings", "test",
        ["unseen_case", "outside_training_regime"])

    # A matching twenty-case roundjet design: same case-level split sizes and
    # analogous in-range, interpolation, operation, temporal, and spatial tests.
    train_jets = (
        (1.0, .75, 1/6, 2.0), (1.0, 1.0, 1/8, 2.4), (1.0, 1.25, 1/6, 3.0),
        (2.0, .75, 1/8, 2.0), (2.0, 1.0, 1/6, 3.0), (2.0, 1.25, 1/8, 2.4),
        (1.0, .75, 1/8, 3.0), (2.0, .75, 1/6, 2.4), (1.0, 1.25, 1/8, 2.0))
    validate_jets = (
        (1.0, 1.0, 1/6, 3.0), (1.0, 1.25, 1/8, 2.4),
        (2.0, .75, 1/6, 3.0), (2.0, 1.0, 1/8, 2.0))
    test_jets = (
        (1.0, 1.0, 1/6, 2.0), (2.0, 1.25, 1/8, 3.0))
    jet_id(D, U, dx, overlap) = "jet_D$(safe_tag(D))_U$(safe_tag(U))_dx$(safe_tag(dx))_ov$(safe_tag(overlap))"
    for (sets, split) in ((train_jets, "train"), (validate_jets, "validate"), (test_jets, "test"))
        for (D, U, dx, overlap) in sets
            plan_case!(jet_id(D, U, dx, overlap), "roundjet", split,
                [split == "test" ? "in_regime" : split])
        end
    end
    for (id, D, U, dx, overlap, Re, thetaod) in (
        ("jet_D1p000_U0p750_dx0p167_ov2p000_Re15000", 1.0, .75, 1/6, 2.0, 15000.0, 0.025),
        ("jet_D1p000_U1p000_dx0p167_ov2p400_theta0p040", 1.0, 1.0, 1/6, 2.4, ROUNDJET_RE, 0.040),
        ("jet_D1p000_U1p500_dx0p167_ov2p400", 1.0, 1.5, 1/6, 2.4, ROUNDJET_RE, 0.025),
        ("jet_D1p000_U1p000_dx0p167_ov2p400_temporal", 1.0, 1.0, 1/6, 2.4, ROUNDJET_RE, 0.025),
        ("jet_D2p000_U1p000_dx0p125_ov2p400_spatial", 2.0, 1.0, 1/8, 2.4, ROUNDJET_RE, 0.025))
        categories = occursin("Re15000", id) ? ["unseen_operating_condition", "unseen_Reynolds_number"] :
            occursin("theta0p040", id) ? ["unseen_boundary_condition", "unseen_operating_condition"] :
            occursin("temporal", id) ? ["temporal_superresolution", "unseen_operating_condition"] :
            occursin("spatial", id) ? ["spatial_superresolution"] :
            ["outside_training_regime", "unseen_operating_condition"]
        plan_case!(id, "roundjet", "test", categories;
            steps=occursin("temporal", id) ? 180 : round(Int, ROUNDJET_TRAVEL_D * ROUNDJET_STEPS_PER_D),
            save_every=occursin("temporal", id) ? 1 : SAVE_EVERY,
            grid_points=occursin("spatial", id) ? 64 : GRID_POINTS,
            Reynolds_number=Re, diameter=D, exit_speed=U,
            momentum_thickness_over_diameter=thetaod,
            particle_spacing_over_momentum_thickness=dx, particle_overlap=overlap)
    end

    # Preserve selected folders' split assignments and metadata. Discarded IDs
    # are intentionally omitted from the active manifest; raw deletion is done
    # separately from this planner.
    existing = Dict{String,Dict{String,Any}}()
    for split in ("train", "test", "validate", "validation"), sim in ("vortexrings", "roundjet")
        dir = joinpath(OUTPUT_ROOT, split, sim)
        isdir(dir) || continue
        for case_dir in readdir(dir; join=true)
            isdir(case_dir) || continue
            toml_path = joinpath(case_dir, "case.toml")
            metadata = isfile(toml_path) ? TOML.parsefile(toml_path) : Dict{String,Any}()
            existing[String(get(metadata, "case_id", basename(case_dir)))] = metadata
        end
    end
    for (id, plan) in CASE_PLAN
        haskey(existing, id) || continue
        old_split = String(get(existing[id], "split", plan["split"]))
        old_split == "validation" && (old_split = "validate")
        old_split == plan["split"] || error("Existing case $id is assigned to $old_split, planned for $(plan["split"])")
        # All selected cases use a uniform 4-step cadence and 720-step standard
        # horizon (180 transitions). The temporal test intentionally saves each
        # of its first 180 integration steps.
        plan["save_every"] = Int(plan["save_every"])
        plan["steps"] = Int(plan["steps"])
    end

    counts = Dict((sim, split) => count(p -> p["simulation"] == sim && p["split"] == split, values(CASE_PLAN))
        for sim in ("vortexrings", "roundjet") for split in ("train", "validate", "test"))
    counts[("vortexrings", "train")] == 9 && counts[("vortexrings", "validate")] == 4 && counts[("vortexrings", "test")] == 7 || error("Vortex-ring plan must contain 9/4/7 train/validate/test cases")
    counts[("roundjet", "train")] == 9 && counts[("roundjet", "validate")] == 4 && counts[("roundjet", "test")] == 7 || error("Roundjet plan must contain 9/4/7 train/validate/test cases")
    manifest_cases = [Dict("case_id" => id, "simulation" => plan["simulation"],
        "split" => plan["split"], "evaluation_categories" => plan["categories"],
        "save_every_steps" => plan["save_every"], "grid_points" => plan["grid_points"],
        "steps" => plan["steps"]) for (id, plan) in sort!(collect(CASE_PLAN); by=first)]
    open(joinpath(OUTPUT_ROOT, "case_plan.toml"), "w") do io
        TOML.print(io, Dict("case_budget" => length(CASE_PLAN), "split_counts" => Dict(
            "vortexrings" => Dict(split => counts[("vortexrings", split)] for split in ("train", "validate", "test")),
            "roundjet" => Dict(split => counts[("roundjet", split)] for split in ("train", "validate", "test"))),
            "transitions_per_standard_case" => 180, "cases" => manifest_cases))
    end
    println("[plan] $(length(CASE_PLAN)) cases: vortexrings 20, roundjet 20; split counts=$counts")
end

function run_vortexrings_sweep()
    case_index = 0
    case_limit = CASE_LIMIT > 0 ? CASE_LIMIT : typemax(Int)

    # Mid/high particle counts expand the image-defined leapfrogging sweep.
    for N in (160, 320, 640), dR in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5), ratio in (1.5, 1.0, 0.75, 0.5, 0.25)
        case_index += 1
        if case_index > case_limit
            return
        end
        case_id = "leapfrog_N$(N)_d$(safe_tag(dR))_g$(safe_tag(ratio))"
        haskey(CASE_PLAN, case_id) || continue
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
        haskey(CASE_PLAN, case_id) || continue
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
        haskey(CASE_PLAN, case_id) || continue
        run_vortex_case(case_id,
            Dict("family" => "signed_two_rings", "total_particles" => N,
                 "radius" => 1.0, "spacing_over_radius" => dR,
                 "circulation_ratio" => [1.0, ratio]),
            2, [10.0, 10.0 * ratio], [1.0, 1.0], N, dR)
    end

    # Toroidal-vorticity ring using FLOWVPM's Cartesian-grid initializer.
    # Preserve the example's Gaussian core, Rcross/R=0.1, sigma=Rcross,
    # and 5% vorticity cutoff while sweeping circulation, radius, and dx.
    for radius in (0.75, 1.0, 1.25, 1.5), gamma in (5.0, 10.0, 20.0), dxratio in (0.5, 0.75, 1.0)
        case_index += 1
        case_index > case_limit && return
        case_id = "toroidal_R$(safe_tag(radius))_g$(safe_tag(gamma))_dx$(safe_tag(dxratio))"
        haskey(CASE_PLAN, case_id) || continue
        run_toroidal_case(radius, gamma, dxratio)
    end

    # Freely evolving pair in unbounded space, with controlled center spacing,
    # circulation ratio, and particle count. The pair is laterally offset.
    for N in (160, 320, 640), separationR in (2.5, 3.0, 4.0, 5.0), ratio in (-1.0, -0.75, -0.5)
        case_index += 1
        case_index > case_limit && return
        case_id = "free_pair_N$(N)_sep$(safe_tag(separationR))_g$(safe_tag(ratio))"
        haskey(CASE_PLAN, case_id) || continue
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
        case_id = "perturbed_N$(N)_amp$(safe_tag(amplitudeR))_mode$(mode)"
        haskey(CASE_PLAN, case_id) || continue
        run_perturbed_ring_case(N, amplitudeR, mode)
    end

    # Matched-parameter high-resolution target for a clean spatial
    # superresolution evaluation against the N=320, d/R=1, ratio=1 train case.
    case_index += 1
    case_index > case_limit && return
    case_id = "leapfrog_spatial_N320_d1p000_g1p000"
    haskey(CASE_PLAN, case_id) && run_vortex_case(case_id,
        Dict("family" => "leapfrogging_spatial_superresolution", "total_particles" => 320,
            "radius" => 1.0, "spacing_over_radius" => 1.0,
            "circulation_ratio" => [1.0, 1.0]),
        2, [10.0, 10.0], [1.0, 1.0], 320, 1.0)
end

function run_toroidal_case(radius, gamma, dxratio)
    case_id = "toroidal_R$(safe_tag(radius))_g$(safe_tag(gamma))_dx$(safe_tag(dxratio))"
    haskey(CASE_PLAN, case_id) || return
    metadata = Dict("family" => "toroidal_vorticity", "radius" => radius,
        "circulation" => gamma, "aspect_ratio" => 1.0,
        "core_radius_over_radius" => 0.10, "grid_spacing_over_core_radius" => dxratio,
        "smoothing_radius_over_core_radius" => 1.0, "minWfraction" => 0.05)
    case_dir = new_case_dir("vortexrings", case_id, metadata)
    case_dir === nothing && return
    fmm = generation_fmm()
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
        nsteps_save=case_save_every(case_id))

    pmin, pmax = vortex_field_bounds([[0.0, 0.0, 0.0]], [radius], [gamma], VORTEX_RTOT)
    metadata["field_bounds_min"] = pmin
    metadata["field_bounds_max"] = pmax
    record_case_metadata!(case_dir, metadata)
    reconstruct_fields!(pfield, case_dir, "pfield", pmin, pmax)
end

function run_perturbed_ring_case(N, amplitudeR, mode)
    case_id = "perturbed_N$(N)_amp$(safe_tag(amplitudeR))_mode$(mode)"
    haskey(CASE_PLAN, case_id) || return
    metadata = Dict("family" => "three_dimensional_perturbation", "total_particles" => N,
        "radius" => 1.0, "circulation" => 10.0, "core_radius_over_radius" => 0.10,
        "perturbation_amplitude_over_radius" => amplitudeR,
        "azimuthal_mode" => mode)
    case_dir = new_case_dir("vortexrings", case_id, metadata)
    case_dir === nothing && return
    fmm = generation_fmm()
    solver = (formulation=vpm.cVPM, SFS=vpm.noSFS,
        relaxation=vpm.correctedpedrizzetti, kernel=vpm.winckelmans,
        viscous=vpm.Inviscid(), transposed=true, integration=vpm.rungekutta3,
        UJ=vpm.UJ_fmm, fmm=fmm)
    pfield = vpm.ParticleField(N + case_grid_points(case_id)^3; solver...)
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
    nsteps = Int(get(CASE_PLAN[case_id], "steps", VORTEX_STEPS))
    Rtot = VORTEX_RTOT
    # Use the standard timestep and retain only the initial 120 steps. The
    # every-step output then provides true temporal refinement of early flow.
    dt = (Rtot / Uref) / VORTEX_STEPS
    println("\n[$case_id] held-out three-dimensional ring perturbation")
    VortexRingCode.run_vortexring_simulation(pfield, nsteps, dt, 1,
        [Nphi], [1], [0];
        save_path=case_dir, run_name="pfield", prompt=false,
        calc_monitors=false, verbose=true, verbose_nsteps=100,
        nsteps_save=case_save_every(case_id))

    pmin, pmax = vortex_field_bounds([[0.0, 0.0, 0.0]], [R], [10.0], Rtot)
    metadata["field_bounds_min"] = pmin
    metadata["field_bounds_max"] = pmax
    record_case_metadata!(case_dir, metadata)
    reconstruct_fields!(pfield, case_dir, "pfield", pmin, pmax)
end

function run_roundjet_case(case_id, metadata, diameter, exit_speed, dxotheta, overlap)
    case_dir = new_case_dir("roundjet", case_id, metadata)
    case_dir === nothing && return
    fmm = generation_fmm()
    case_steps = Int(get(CASE_PLAN[case_id], "steps", round(Int, ROUNDJET_TRAVEL_D * ROUNDJET_STEPS_PER_D)))
    travel_d = case_steps / ROUNDJET_STEPS_PER_D
    thetaod = Float64(get(metadata, "momentum_thickness_over_diameter", 0.025))
    maxparticles = roundjet_particle_capacity(diameter, exit_speed, dxotheta, overlap,
        case_grid_points(case_id), thetaod)
    viscous = vpm.CoreSpreading(-1.0, -1.0, RoundJetCode.zeta_fmm_compat;
        itmax=20, tol=1e-2, iterror=false)

    println("\n[$case_id] D=$diameter, Uexit=$exit_speed, dx/theta=$dxotheta, overlap=$overlap")
    metadata["Reynolds_number"] = Float64(get(metadata, "Reynolds_number", ROUNDJET_RE))
    metadata["steps_per_diameter"] = ROUNDJET_STEPS_PER_D
    metadata["travel_diameters"] = travel_d
    pmin = [-2.5 * diameter, -2.5 * diameter, -2.0 * diameter]
    pmax = [2.5 * diameter, 2.5 * diameter, (travel_d + 2.0) * diameter]
    metadata["field_bounds_min"] = pmin
    metadata["field_bounds_max"] = pmax
    pfield = RoundJetCode.run_roundjet_simulation(diameter, exit_speed, 0.0;
        thetaod=thetaod,
        steps_per_d=ROUNDJET_STEPS_PER_D,
        d_travel_tot=travel_d,
        dxotheta=dxotheta,
        overlap=overlap,
        minWfraction=1e-2,
        maxRoR=1.0,
        rbf=true,
        rbf_optargs=[(:itmax, ROUNDJET_RBF_ITMAX), (:tol, 2.5e-2),
                     (:iterror, true), (:verbose, true), (:debug, false)],
        Re=Float64(get(metadata, "Reynolds_number", ROUNDJET_RE)),
        maxparticles=maxparticles,
        pfieldargs=(fmm=fmm, kernel=vpm.gaussianerf, viscous=viscous),
        save_path=case_dir,
        run_name="pfield",
        prompt=false,
        verbose=true,
        verbose_nsteps=100,
        nsteps_save=case_save_every(case_id))

    metadata["final_particle_count"] = vpm.get_np(pfield)
    metadata["completed_steps"] = pfield.nt
    record_case_metadata!(case_dir, metadata)
    reconstruct_fields!(pfield, case_dir, "pfield", pmin, pmax)
end

function roundjet_particle_capacity(diameter, exit_speed, dxotheta, overlap, grid_points, thetaod)
    # Size for initialization and field reconstruction. The round-jet driver
    # grows its particle buffer as new inlet particles arrive over the trajectory.
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
    return ceil(Int, 1.25 * initial_particles) + grid_points^3 + 4096
end

function run_roundjet_sweep()
    # The balanced 20-case jet plan samples the same train/validation/test
    # proportions as vortex rings and uses a 7.2D standard trajectory.
    case_index = 0
    case_limit = CASE_LIMIT > 0 ? CASE_LIMIT : typemax(Int)
    for diameter in (1.0, 2.0), exit_speed in (0.75, 1.0, 1.25), dxotheta in (1/6, 1/8), overlap in (2.0, 2.4, 3.0)
        case_index += 1
        case_index > case_limit && return
        case_id = "jet_D$(safe_tag(diameter))_U$(safe_tag(exit_speed))_dx$(safe_tag(dxotheta))_ov$(safe_tag(overlap))"
        haskey(CASE_PLAN, case_id) || continue
        run_roundjet_case(case_id,
            Dict("family" => "roundjet", "diameter" => diameter,
                 "exit_speed" => exit_speed, "coflow_speed" => 0.0,
                 "momentum_thickness_over_diameter" => 0.025,
                 "particle_spacing_over_momentum_thickness" => dxotheta,
                 "particle_overlap" => overlap,
                 "Reynolds_number" => ROUNDJET_RE),
            diameter, exit_speed, dxotheta, overlap)
    end

    for (case_id, diameter, speed, dxotheta, overlap, Re, thetaod) in (
        ("jet_D1p000_U0p750_dx0p167_ov2p000_Re15000", 1.0, 0.75, 1/6, 2.0, 15000.0, 0.025),
        ("jet_D1p000_U1p000_dx0p167_ov2p400_theta0p040", 1.0, 1.0, 1/6, 2.4, ROUNDJET_RE, 0.040),
        ("jet_D1p000_U1p500_dx0p167_ov2p400", 1.0, 1.5, 1/6, 2.4, ROUNDJET_RE, 0.025),
        ("jet_D1p000_U1p000_dx0p167_ov2p400_temporal", 1.0, 1.0, 1/6, 2.4, ROUNDJET_RE, 0.025),
        ("jet_D2p000_U1p000_dx0p125_ov2p400_spatial", 2.0, 1.0, 1/8, 2.4, ROUNDJET_RE, 0.025),
    )
        case_index += 1
        case_index > case_limit && return
        haskey(CASE_PLAN, case_id) || continue
        metadata = Dict("family" => "roundjet", "diameter" => diameter,
            "exit_speed" => speed, "coflow_speed" => 0.0,
            "momentum_thickness_over_diameter" => thetaod,
            "particle_spacing_over_momentum_thickness" => dxotheta,
            "particle_overlap" => overlap, "Reynolds_number" => Re)
        run_roundjet_case(case_id, metadata, diameter, speed, dxotheta, overlap)
    end
end

mkpath(OUTPUT_ROOT)
println("Train/validation/test output: $OUTPUT_ROOT/{train,validate,test}/{vortexrings,roundjet}")
println("Grid: $(GRID_POINTS) x $(GRID_POINTS) x $(GRID_POINTS) nodes; snapshots every $SAVE_EVERY steps")
println("Julia threads: $(Threads.nthreads())")

prepare_case_plan!()
if DATASET in ("all", "vortexrings")
    run_vortexrings_sweep()
end
if DATASET in ("all", "roundjet")
    run_roundjet_sweep()
end

println("Training-data generation complete.")
