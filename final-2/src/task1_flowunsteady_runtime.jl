module Task1FlowUnsteadyRuntime

using JLD2
using Printf

include("task1_ugradu_settings.jl")
include("task1_particle_io.jl")
include("task1_python_predictor.jl")

using .Task1UGradUSettings
using .Task1ParticleIO
using .Task1PythonPredictor

export build_runtime_callback

"""
Build runtime callback to:
- export baseline U/gradU supervision
- or replace U/gradU with ML predictions
- or compare both in one run
"""
function build_runtime_callback(cfg::UGradUSettings)
    mkpath(cfg.save_dir)

    predictor = nothing
    if cfg.mode == :surrogate_ml || cfg.mode == :compare
        predictor = build_predictor(cfg.model_py_path, cfg.model_meta_npz; device=cfg.device)
    end

    function cb(sim, pfield, t, dt; optargs...)
        step = Int(round(t / dt))
        if step % cfg.save_stride != 0
            return true
        end

        X, Gamma, sigma = get_particle_state(pfield)
        feats = build_input_features(
            X, Gamma, sigma;
            use_context_channels=cfg.use_context_channels,
            phase=cfg.phase,
            aoa_deg=cfg.aoa_deg,
            freestream=cfg.freestream,
        )

        # Baseline references before overwrite
        U_ref = copy(pfield.U)'
        Gx_ref = copy(pfield.J[1, :, :])'
        Gy_ref = copy(pfield.J[2, :, :])'
        Gz_ref = copy(pfield.J[3, :, :])'

        if cfg.mode == :surrogate_ml || cfg.mode == :compare
            Y = predictor(feats)
            set_particle_ugradu!(pfield, Y)
        end

        # Save snapshots for debugging/comparison
        if cfg.mode == :baseline_fmm
            fname = @sprintf("baseline_%06d.jld2", step)
            @save joinpath(cfg.save_dir, fname) X Gamma sigma U_ref Gx_ref Gy_ref Gz_ref t dt step
        elseif cfg.mode == :surrogate_ml
            U_ml = copy(pfield.U)'
            Gx_ml = copy(pfield.J[1, :, :])'
            Gy_ml = copy(pfield.J[2, :, :])'
            Gz_ml = copy(pfield.J[3, :, :] )'
            fname = @sprintf("surrogate_%06d.jld2", step)
            @save joinpath(cfg.save_dir, fname) X Gamma sigma U_ml Gx_ml Gy_ml Gz_ml t dt step
        else
            U_ml = copy(pfield.U)'
            Gx_ml = copy(pfield.J[1, :, :])'
            Gy_ml = copy(pfield.J[2, :, :])'
            Gz_ml = copy(pfield.J[3, :, :] )'
            dU = U_ml .- U_ref
            dGx = Gx_ml .- Gx_ref
            dGy = Gy_ml .- Gy_ref
            dGz = Gz_ml .- Gz_ref
            fname = @sprintf("compare_%06d.jld2", step)
            @save joinpath(cfg.save_dir, fname) X Gamma sigma U_ref Gx_ref Gy_ref Gz_ref U_ml Gx_ml Gy_ml Gz_ml dU dGx dGy dGz t dt step
        end

        return true
    end

    return cb
end

end # module
