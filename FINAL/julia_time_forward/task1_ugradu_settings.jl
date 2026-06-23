module Task1UGradUSettings

export UGradUSettings, default_settings

"""
Configuration for Task-1 U/gradU surrogate runtime.

`mode`:
- `:baseline_fmm`  => normal FLOWUnsteady FMM U/gradU
- `:surrogate_ml`  => replace U/gradU with ML predictor
- `:compare`       => run both paths and save differences
"""
Base.@kwdef struct UGradUSettings
    mode::Symbol = :baseline_fmm
    save_dir::String = "final-2/output/runtime"
    save_stride::Int = 1
    model_py_path::String = "final-2/output/task1_ugradu_training/best_task1_ugradu_gno.pt"
    model_meta_npz::String = "final-2/output/particle_ugradu_dataset.npz"
    device::String = "cpu"
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
end

default_settings() = UGradUSettings()

end # module
