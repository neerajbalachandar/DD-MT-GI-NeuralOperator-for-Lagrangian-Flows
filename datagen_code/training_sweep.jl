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
const CASE_BUDGET = parse(Int, get(ENV, "CASE_BUDGET", "120"))
const RESUME = lowercase(get(ENV, "RESUME", "1")) in ("1", "true", "yes")
const NGRID = GRID_POINTS^3
const CASE_PLAN = Dict{String,Dict{String,Any}}()

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
    plan = get(CASE_PLAN, case_id, Dict{String,Any}())
    split = String(get(plan, "split", "train"))
    metadata["evaluation_categories"] = get(plan, "categories", [split == "test" ? "in_regime" : split])
    metadata["split_rule"] = "Balanced case-level plan; all frames from a case stay in one split"
    folder_name = case_id
    if split == "test" && !isempty(get(plan, "categories", String[]))
        category = first(get(plan, "categories", String[]))
        prefix = category == "outside_training_regime" ? "test_outside_regime__" :
            category == "unseen_operating_condition" ? "test_unseen_operation__" :
            category == "unseen_case" ? "test_unseen_initial__" : ""
        isempty(prefix) || (folder_name = prefix * case_id)
    end
    case_dir = joinpath(OUTPUT_ROOT, split, simulation, folder_name)
    metadata["split"] = split
    metadata["simulation"] = simulation
    metadata["case_id"] = case_id
    metadata["grid_points"] = [case_grid_points(case_id), case_grid_points(case_id), case_grid_points(case_id)]
    metadata["save_every_steps"] = case_save_every(case_id)
    metadata["fidelity_profile"] = "mid_high"
    metadata["field_components"] = ["U", "W"]
    if ispath(case_dir)
        if RESUME && case_is_complete(case_dir, metadata)
            println("[resume] complete case already exists; skipping: $case_id")
            old_metadata = isfile(joinpath(case_dir, "case.toml")) ?
                TOML.parsefile(joinpath(case_dir, "case.toml")) : Dict{String,Any}()
            merge!(metadata, old_metadata)
            metadata["evaluation_categories"] = get(plan, "categories", [split == "test" ? "in_regime" : split])
            metadata["split"] = split
            record_case_metadata!(case_dir, metadata)
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
    leapfrog = ["leapfrog_N$(N)_d$(safe_tag(dR))_g$(safe_tag(r))"
                for N in (160, 320, 640), dR in (0.75, 1.0, 1.25, 1.5, 2.0), r in (1.0, 0.75, 0.5)]
    isolated = ["isolated_N$(N)_g$(safe_tag(gamma))_sigma$(safe_tag(sigmaR))"
                for N in (160, 320, 640), gamma in (5.0, 10.0, 20.0), sigmaR in (0.02, 0.04, 0.08)]
    signed = ["signed_N$(N)_d$(safe_tag(dR))_g$(safe_tag(r))"
              for N in (160, 320), dR in (0.75, 1.0, 1.5, 2.0), r in (-1.0, -0.75, -0.5)]
    toroidal = ["toroidal_R$(safe_tag(radius))_g$(safe_tag(gamma))_dx$(safe_tag(dxratio))"
                for radius in (0.75, 1.0, 1.25), gamma in (5.0, 10.0, 20.0), dxratio in (0.5, 0.75, 1.0)]
    freepair = ["free_pair_N$(N)_sep$(safe_tag(separationR))_g$(safe_tag(r))"
                for N in (160, 320, 640), separationR in (2.5, 3.0, 4.0), r in (-1.0, -0.75, -0.5)]
    perturbed = ["perturbed_N$(N)_amp$(safe_tag(amplitudeR))_mode$(mode)"
                 for N in (160, 320, 640), amplitudeR in (0.02, 0.05, 0.10), mode in (2, 3)]
    roundjet = ["jet_D$(safe_tag(diameter))_U$(safe_tag(speed))_dx$(safe_tag(dx))_ov$(safe_tag(overlap))"
                for diameter in (1.0, 2.0), speed in (0.75, 1.0, 1.25), dx in (1/6, 1/8), overlap in (2.0, 2.4, 3.0)]

    # Keep every completed case already present on the requested drive and
    # include the interrupted 24th leapfrog configuration for regeneration.
    existing = Dict{String,Dict{String,Any}}()
    for split in ("train", "test", "validate", "validation"), simulation in ("vortexrings", "roundjet")
        dir = joinpath(OUTPUT_ROOT, split, simulation)
        isdir(dir) || continue
        for case_dir in readdir(dir; join=true)
            isdir(case_dir) || continue
            toml_path = joinpath(case_dir, "case.toml")
            metadata = isfile(toml_path) ? TOML.parsefile(toml_path) : Dict{String,Any}("split" => split)
            existing[String(get(metadata, "case_id", basename(case_dir)))] = metadata
        end
    end
    kept_leap = String[id for id in leapfrog if haskey(existing, id)]
    chosen_leap = evenly_select(leapfrog, 27;
        required=unique(vcat(kept_leap, ["leapfrog_N320_d1p250_g0p500"])))
    chosen_toroid = String[]
    for radius in (0.75, 1.0, 1.25)
        group = filter(id -> startswith(id, "toroidal_R$(safe_tag(radius))_"), toroidal)
        append!(chosen_toroid, evenly_select(group, radius == 1.0 ? 6 : 5))
    end
    chosen_roundjet = evenly_select(roundjet, 20)
    unseen_re_cases = ["jet_D1p000_U0p750_dx0p167_ov2p000_Re15000",
                       "jet_D2p000_U1p250_dx0p125_ov3p000_Re15000"]
    chosen_perturbed = evenly_select(perturbed, 9)
    selected = vcat(chosen_leap, evenly_select(isolated, 16), evenly_select(signed, 14),
                    chosen_toroid, evenly_select(freepair, 16), chosen_perturbed,
                    chosen_roundjet, unseen_re_cases)
    length(selected) == CASE_BUDGET || error("Balanced plan has $(length(selected)) cases, CASE_BUDGET=$CASE_BUDGET")
    CASE_BUDGET == 120 || error("This balanced parameter design is defined for CASE_BUDGET=120")

    # The two added Reynolds-number cases are held out operation conditions.
    test_categories = Dict{String,Vector{String}}()
    for id in chosen_toroid
        occursin("toroidal_R1p250_", id) && (test_categories[id] = ["outside_training_regime"])
    end
    test_categories[unseen_re_cases[1]] = ["unseen_operating_condition", "unseen_Reynolds_number"]
    test_categories[unseen_re_cases[2]] = ["unseen_operating_condition", "unseen_Reynolds_number", "spatial_superresolution"]
    for id in chosen_perturbed
        test_categories[id] = ["unseen_case", "unseen_initial_condition"]
    end
    test_categories[first(chosen_perturbed)] = ["unseen_case", "unseen_initial_condition", "temporal_superresolution"]

    for (id, meta) in existing
        haskey(CASE_PLAN, id) && continue
        # Existing cases are retained only when they belong to the chosen plan.
        id in selected || continue
        split = String(get(meta, "split", "train"))
        cats = String.(get(meta, "evaluation_categories", [split == "test" ? "in_regime" : split]))
        CASE_PLAN[id] = Dict{String,Any}("split" => (split == "validation" ? "validate" : split),
            "categories" => cats, "save_every" => Int(get(meta, "save_every_steps", SAVE_EVERY)),
            "grid_points" => Int(get(meta, "grid_points", [GRID_POINTS])[1]))
    end

    new_ids = [id for id in selected if !haskey(CASE_PLAN, id)]
    mandatory_test = Set(vcat(collect(keys(test_categories)), unseen_re_cases))
    for id in chosen_perturbed
        push!(mandatory_test, id)
    end
    for id in mandatory_test
        id in new_ids || continue
        cats = get(test_categories, id, ["unseen_operating_condition"])
        CASE_PLAN[id] = Dict{String,Any}("split" => "test", "categories" => cats,
            "save_every" => id == first(chosen_perturbed) ? 1 : SAVE_EVERY,
            "grid_points" => id == unseen_re_cases[2] ? 64 : GRID_POINTS)
    end
    current_train = count(p -> get(p, "split", "") == "train", values(CASE_PLAN))
    current_val = count(p -> get(p, "split", "") == "validate", values(CASE_PLAN))
    current_test = count(p -> get(p, "split", "") == "test", values(CASE_PLAN))
    remaining = [id for id in new_ids if !haskey(CASE_PLAN, id)]
    # Fill any remaining test slots with in-regime cases. On the interrupted
    # run, the two existing test cases already fill these slots; on a clean
    # output root, this selects two cases inside the training parameter range.
    need_test = 18 - current_test
    need_test >= 0 || error("Existing test cases exceed the planned budget")
    length(remaining) >= need_test || error("Not enough cases to fill the in-regime test split")
    for id in remaining[1:need_test]
        CASE_PLAN[id] = Dict{String,Any}("split" => "test", "categories" => ["in_regime"],
            "save_every" => SAVE_EVERY, "grid_points" => GRID_POINTS)
    end
    remaining = remaining[need_test+1:end]
    need_train, need_val = 84 - current_train, 18 - current_val
    need_train >= 0 && need_val >= 0 || error("Existing cases exceed the planned train/validate budget")
    length(remaining) == need_train + need_val ||
        error("Balanced plan needs $(need_train + need_val) train/validate cases but has $(length(remaining))")
    # Stratify validation across flow families so it represents both the
    # vortex-ring variants and roundjet conditions in the same case ratio.
    family_prefixes = ("leapfrog_", "isolated_", "signed_", "toroidal_",
                       "free_pair_", "perturbed_", "jet_")
    family_groups = [filter(id -> startswith(id, prefix), remaining) for prefix in family_prefixes]
    val_ids = Set{String}()
    cursors = ones(Int, length(family_groups))
    while length(val_ids) < need_val
        progressed = false
        for (index, group) in enumerate(family_groups)
            if cursors[index] <= length(group)
                push!(val_ids, group[cursors[index]])
                cursors[index] += 1
                progressed = true
                length(val_ids) == need_val && break
            end
        end
        progressed || error("Could not stratify the requested validation case count")
    end
    for id in remaining
        split = id in val_ids ? "validate" : "train"
        CASE_PLAN[id] = Dict{String,Any}("split" => split,
            "categories" => [split], "save_every" => SAVE_EVERY, "grid_points" => GRID_POINTS)
    end
    final_counts = Dict(split => count(p -> p["split"] == split, values(CASE_PLAN))
                        for split in ("train", "validate", "test"))
    final_counts == Dict("train" => 84, "validate" => 18, "test" => 18) ||
        error("Selected case split counts are not balanced: $final_counts")
    manifest_cases = [Dict("case_id" => id,
        "simulation" => (startswith(id, "jet_") ? "roundjet" : "vortexrings"),
        "split" => CASE_PLAN[id]["split"], "evaluation_categories" => CASE_PLAN[id]["categories"],
        "save_every_steps" => CASE_PLAN[id]["save_every"], "grid_points" => CASE_PLAN[id]["grid_points"])
        for id in sort!(collect(keys(CASE_PLAN)))]
    open(joinpath(OUTPUT_ROOT, "case_plan.toml"), "w") do io
        TOML.print(io, Dict("case_budget" => CASE_BUDGET, "split_counts" => final_counts,
                            "max_standard_pairs_per_case" => 100, "cases" => manifest_cases))
    end
    println("[plan] $(length(CASE_PLAN)) cases: $(final_counts); preserving $(length(kept_leap)) existing leapfrog cases")
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
    for radius in (0.75, 1.0, 1.25), gamma in (5.0, 10.0, 20.0), dxratio in (0.5, 0.75, 1.0)
        case_index += 1
        case_index > case_limit && return
        case_id = "toroidal_R$(safe_tag(radius))_g$(safe_tag(gamma))_dx$(safe_tag(dxratio))"
        haskey(CASE_PLAN, case_id) || continue
        run_toroidal_case(radius, gamma, dxratio)
    end

    # Freely evolving pair in unbounded space, with controlled center spacing,
    # circulation ratio, and particle count. The pair is laterally offset.
    for N in (160, 320, 640), separationR in (2.5, 3.0, 4.0), ratio in (-1.0, -0.75, -0.5)
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
    for (case_id, diameter, speed, dxotheta, overlap) in (
        ("jet_D1p000_U0p750_dx0p167_ov2p000_Re15000", 1.0, 0.75, 1/6, 2.0),
        ("jet_D2p000_U1p250_dx0p125_ov3p000_Re15000", 2.0, 1.25, 1/8, 3.0),
    )
        metadata = Dict("family" => "roundjet", "diameter" => diameter,
            "exit_speed" => speed, "coflow_speed" => 0.0,
            "momentum_thickness_over_diameter" => 0.025,
            "particle_spacing_over_momentum_thickness" => dxotheta,
            "particle_overlap" => overlap, "Reynolds_number" => 15000.0)
        run_roundjet_case(case_id, metadata, diameter, speed, dxotheta, overlap)
    end
end

mkpath(OUTPUT_ROOT)
println("Train/validation/test output: $OUTPUT_ROOT/{train,validate,test}")
println("Grid: $(GRID_POINTS) x $(GRID_POINTS) x $(GRID_POINTS) nodes; snapshots every $SAVE_EVERY steps")

if DATASET in ("all", "vortexrings")
    prepare_case_plan!()
    run_vortexrings_sweep()
end
if DATASET in ("all", "roundjet")
    isempty(CASE_PLAN) && prepare_case_plan!()
    run_roundjet_sweep()
end

println("Training-data generation complete.")
