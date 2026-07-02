"""
Runner for Task-1 U/gradU integration with FLOWUnsteady, without package edits.

Usage (example):
  julia --project=/path/to/FLOWUnsteady run_task1_flowunsteady.jl \
    --sim-file simulation.jl \
    --mode compare \
    --save-dir output/runtime_compare

This script expects your sim file to define a function:
  build_task1_simulation() -> (sim, run_kwargs::Dict)
where `run_kwargs` are forwarded to `FLOWUnsteady.run_simulation`.
"""

include("task1_ugradu_settings.jl")
include("flowunsteady_callback.jl")

using .Task1UGradUSettings
using .Task1FlowUnsteadyRuntime

const THIS_DIR = @__DIR__
const FINAL_DIR = normpath(joinpath(THIS_DIR, ".."))
const DEFAULT_TASK1_MODEL = joinpath(FINAL_DIR, "result", "task1", "best_task1_v3_model.pt")
const DEFAULT_TASK1_META = joinpath(FINAL_DIR, "processed_data_task1", "particle_ugradu_dataset.npz")

function resolve_from_workflow(path::AbstractString)
    expanded = expanduser(String(path))
    if isabspath(expanded)
        return normpath(expanded)
    end

    workflow_path = normpath(joinpath(THIS_DIR, expanded))
    if ispath(workflow_path) || startswith(expanded, "output")
        return workflow_path
    end

    return abspath(expanded)
end

function parse_cli()
    defaults = Dict{String,String}(
        "mode" => "baseline_fmm",
        "save_dir" => joinpath(THIS_DIR, "output", "runtime"),
        "model" => DEFAULT_TASK1_MODEL,
        "meta" => DEFAULT_TASK1_META,
        "device" => "auto",
    )
    if any(arg -> arg == "--help" || arg == "-h", ARGS)
        println("""
        Usage:
          julia --project=<FLOWUnsteady-env> /home/dysco/FLOWUnsteady/Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/run_task1_flowunsteady.jl \\
            --sim-file simulation.jl [options]

        Options:
          --sim-file PATH   Required. Julia file defining build_task1_simulation()
          --mode MODE       baseline_fmm | surrogate_ml | compare
          --save-dir DIR    Runtime snapshot/output directory
          --model PATH      Task1 U/gradU .pt checkpoint
          --meta PATH       particle_ugradu_dataset.npz metadata
          --device DEVICE   auto | cpu | cuda | cuda:<id>
        """)
        exit(0)
    end

    args = copy(defaults)
    i = 1
    while i <= length(ARGS)
        key = ARGS[i]
        if !startswith(key, "--")
            error("Unexpected positional argument: $(key)")
        end
        name = replace(key[3:end], "-" => "_")
        if i == length(ARGS)
            error("Missing value for $(key)")
        end
        args[name] = ARGS[i + 1]
        i += 2
    end
    if !haskey(args, "sim_file")
        error("Missing required --sim-file")
    end
    return args
end

function main()
    args = parse_cli()

    simfile = resolve_from_workflow(args["sim_file"])
    include(simfile)
    if !@isdefined build_task1_simulation
        error("Sim file must define build_task1_simulation()")
    end

    mode_sym = Symbol(args["mode"])
    save_dir = resolve_from_workflow(args["save_dir"])
    flowunsteady_output_dir = joinpath(THIS_DIR, "output", "flowunsteady_$(mode_sym)")
    sim, run_kwargs = Base.invokelatest(
        build_task1_simulation;
        output_dir=flowunsteady_output_dir,
        run_label="wing-example_$(mode_sym)",
    )

    model_path = resolve_from_workflow(args["model"])
    meta_path = resolve_from_workflow(args["meta"])
    if (mode_sym == :surrogate_ml || mode_sym == :compare) && !isfile(model_path)
        error("Task1 model checkpoint not found: $(model_path)")
    end
    if (mode_sym == :surrogate_ml || mode_sym == :compare) && !isfile(meta_path)
        error("Task1 metadata dataset not found: $(meta_path)")
    end

    cfg = UGradUSettings(
        mode = mode_sym,
        save_dir = save_dir,
        model_py_path = model_path,
        model_meta_npz = meta_path,
        device = args["device"],
    )

    predictor = nothing
    predict_in_callback = true
    if mode_sym == :surrogate_ml
        ml_uj, predictor, _close_predictor = build_ml_uj_function(cfg)
        run_kwargs[:vpm_UJ] = ml_uj
        predict_in_callback = false
        println("Task1 surrogate_ml: replacing FLOWVPM UJ/FMM with persistent ML U/gradU predictor")
    end

    cb = build_runtime_callback(cfg; predictor=predictor, predict_in_callback=predict_in_callback)

    if !haskey(run_kwargs, :nsteps)
        error("run_kwargs must include :nsteps")
    end

    existing_cb = get(run_kwargs, :extra_runtime_function, nothing)
    if existing_cb === nothing
        run_kwargs[:extra_runtime_function] = cb
    else
        function combined_runtime_callback(args...; kwargs...)
            should_stop = existing_cb(args...; kwargs...)
            should_stop === true && return true
            return cb(args...; kwargs...)
        end
        run_kwargs[:extra_runtime_function] = combined_runtime_callback
    end

    nsteps = pop!(run_kwargs, :nsteps)
    run_kwargs[:prompt] = false
    @eval import FLOWUnsteady
    Base.invokelatest(FLOWUnsteady.run_simulation, sim, nsteps; run_kwargs...)
end

main()
