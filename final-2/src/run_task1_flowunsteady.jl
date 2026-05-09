"""
Runner for Task-1 U/gradU integration with FLOWUnsteady, without package edits.

Usage (example):
  julia --project run_task1_flowunsteady.jl \
    --sim-file /path/to/your/simulation_definition_wing.jl \
    --mode compare \
    --save-dir final-2/output/runtime_compare

This script expects your sim file to define a function:
  build_task1_simulation() -> (sim, run_kwargs::Dict)
where `run_kwargs` are forwarded to `FLOWUnsteady.run_simulation`.
"""

using ArgParse
using FLOWUnsteady

include("task1_ugradu_settings.jl")
include("task1_flowunsteady_runtime.jl")

using .Task1UGradUSettings
using .Task1FlowUnsteadyRuntime

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--sim-file"
        help = "Path to user simulation file that defines build_task1_simulation()"
        arg_type = String
        required = true

        "--mode"
        help = "baseline_fmm | surrogate_ml | compare"
        arg_type = String
        default = "baseline_fmm"

        "--save-dir"
        help = "Runtime output directory"
        arg_type = String
        default = "final-2/output/runtime"

        "--model"
        help = "Path to trained Task1-u/gradU model .pt"
        arg_type = String
        default = "final-2/output/task1_ugradu_training/best_task1_ugradu_gno.pt"

        "--meta"
        help = "Path to particle_ugradu_dataset.npz"
        arg_type = String
        default = "final-2/output/particle_ugradu_dataset.npz"

        "--device"
        help = "cpu | cuda"
        arg_type = String
        default = "cpu"
    end
    return parse_args(s)
end

function main()
    args = parse_cli()

    simfile = args["sim_file"]
    include(simfile)
    if !@isdefined build_task1_simulation
        error("Sim file must define build_task1_simulation()")
    end

    sim, run_kwargs = build_task1_simulation()

    mode_sym = Symbol(args["mode"])
    cfg = UGradUSettings(
        mode = mode_sym,
        save_dir = args["save_dir"],
        model_py_path = args["model"],
        model_meta_npz = args["meta"],
        device = args["device"],
    )

    cb = build_runtime_callback(cfg)

    # Attach callback/monitor to your run kwargs.
    # Depending on your FLOWUnsteady version, this key can be
    # `extra_runtime_function` or part of monitor composition.
    run_kwargs[:extra_runtime_function] = cb

    if !haskey(run_kwargs, :nsteps)
        error("run_kwargs must include :nsteps")
    end
    FLOWUnsteady.run_simulation(sim, run_kwargs[:nsteps]; run_kwargs...)
end

main()
