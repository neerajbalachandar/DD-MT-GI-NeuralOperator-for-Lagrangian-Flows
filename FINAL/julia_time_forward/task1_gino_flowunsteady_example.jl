"""
Example: use the trained particle_ugradu GINO checkpoint inside a FLOWUnsteady
run, replacing the expensive U/gradU particle query while leaving the standard
rVPM time integrator in control of particle evolution.

Your simulation file should define:
    build_task1_simulation() -> (sim, run_kwargs::Dict)

Run from the repository root, adapting paths as needed:

    julia --project=final-2/src final-2/src/task1_gino_flowunsteady_example.jl \
        /path/to/my_flowunsteady_case.jl
"""

using FLOWUnsteady

include("task1_ugradu_settings.jl")
include("task1_flowunsteady_runtime.jl")

using .Task1UGradUSettings
using .Task1FlowUnsteadyRuntime

if length(ARGS) < 1
    error("Pass a simulation file that defines build_task1_simulation().")
end

include(ARGS[1])
if !@isdefined build_task1_simulation
    error("Simulation file must define build_task1_simulation()")
end

sim, run_kwargs = build_task1_simulation()

cfg = UGradUSettings(
    mode = :surrogate_ml,
    save_dir = "final-2/output/runtime_gino",
    model_py_path = "final-2/result/task1/task1_particle_ugradu_gino_best_model.pt",
    model_meta_npz = "final-2/processed_data_task1/particle_ugradu_dataset.npz",
    device = "cpu",
    input_feature_names = [
        "x", "y", "z",
        "Gamma_x", "Gamma_y", "Gamma_z",
        "sigma",
        "geom_dist", "geom_nx", "geom_ny", "geom_nz",
        "angle_of_attack", "phase",
    ],
    use_context_channels = true,
    aoa_deg = 10.0,
    freestream = (10.0, 0.0, 0.0),
    phase = 0.0,
)

ml_callback = build_runtime_callback(cfg)

# Depending on FLOWUnsteady version, this hook name may be part of a monitor
# composition instead. This repo's runtime wrapper expects the callback to be
# called as cb(sim, pfield, t, dt; optargs...).
run_kwargs[:extra_runtime_function] = ml_callback

if !haskey(run_kwargs, :nsteps)
    error("run_kwargs must include :nsteps")
end

FLOWUnsteady.run_simulation(sim, run_kwargs[:nsteps]; run_kwargs...)
