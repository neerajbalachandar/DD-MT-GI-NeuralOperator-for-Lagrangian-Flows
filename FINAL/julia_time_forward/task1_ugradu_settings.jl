module Task1UGradUSettings

export UGradUSettings, default_settings

const THIS_DIR = @__DIR__
const FINAL_DIR = normpath(joinpath(THIS_DIR, ".."))
const DEFAULT_TASK1_MODEL = joinpath(FINAL_DIR, "result", "task1", "best_task1_v3_model.pt")
const DEFAULT_TASK1_META = joinpath(FINAL_DIR, "processed_data_task1", "particle_ugradu_dataset.npz")

"""
Configuration for Task-1 U/gradU surrogate runtime.

`mode`:
- `:baseline_fmm`  => normal FLOWUnsteady FMM U/gradU
- `:surrogate_ml`  => replace U/gradU with ML predictor
- `:compare`       => run both paths and save differences
"""
Base.@kwdef struct UGradUSettings
    mode::Symbol = :baseline_fmm
    save_dir::String = joinpath(THIS_DIR, "output", "runtime")
    save_stride::Int = 1
    model_py_path::String = DEFAULT_TASK1_MODEL
    model_meta_npz::String = DEFAULT_TASK1_META
    device::String = "auto"
    input_feature_names::Vector{String} = [
        "x", "y", "z",
        "Gamma_x", "Gamma_y", "Gamma_z",
        "sigma",
        "geom_dist", "geom_nx", "geom_ny", "geom_nz",
        "angle_of_attack", "phase",
    ]
    use_context_channels::Bool = true
    aoa_deg::Float64 = 0.0
    freestream::NTuple{3,Float64} = (0.0, 0.0, 0.0)
    phase::Float64 = 0.0
    cache_ml_uj_per_step::Bool = true
end

default_settings() = UGradUSettings()

end # module
