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
    use_context_channels::Bool = false
    aoa_deg::Float64 = 0.0
    freestream::NTuple{3,Float64} = (0.0, 0.0, 0.0)
    phase::Float64 = 0.0
end

default_settings() = UGradUSettings()

end # module
