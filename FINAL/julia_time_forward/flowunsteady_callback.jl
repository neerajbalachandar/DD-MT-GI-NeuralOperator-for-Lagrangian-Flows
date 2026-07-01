module Task1FlowUnsteadyRuntime

using DelimitedFiles
using Printf

include("task1_particle_io.jl")
include("task1_python_predictor.jl")

using .Task1ParticleIO
using .Task1PythonPredictor

export build_runtime_callback

function _save_snapshot_csv(fpath::String, columns::Vector{String}, arrays...)
    data = hcat(arrays...)
    open(fpath, "w") do io
        println(io, join(columns, ","))
        writedlm(io, data, ',')
    end
end

"""
Build runtime callback to:
- export baseline U/gradU supervision
- or replace U/gradU with ML predictions
- or compare both in one run
"""
function build_runtime_callback(cfg)
    mkpath(cfg.save_dir)
    println("Task1 runtime callback active: mode=$(cfg.mode), save_dir=$(cfg.save_dir)")

    predictor = nothing
    if cfg.mode == :surrogate_ml || cfg.mode == :compare
        predictor = build_predictor(cfg.model_py_path, cfg.model_meta_npz; device=cfg.device)
    end

    function cb(sim, pfield, t, dt; optargs...)
        step = Int(round(t / dt))
        if step % cfg.save_stride != 0
            return false
        end

        X, Gamma, sigma = get_particle_state(pfield)
        if size(X, 1) == 0
            return false
        end
        feats = build_input_features(
            X, Gamma, sigma;
            feature_names=cfg.input_feature_names,
            use_context_channels=cfg.use_context_channels,
            phase=cfg.phase,
            aoa_deg=cfg.aoa_deg,
            freestream=cfg.freestream,
        )

        # Baseline references before overwrite
        U_ref, Gx_ref, Gy_ref, Gz_ref = get_particle_ugradu(pfield)

        if cfg.mode == :surrogate_ml || cfg.mode == :compare
            Y = predictor(feats)
            set_particle_ugradu!(pfield, Y)
        end

        # Save snapshots for debugging/comparison
        if cfg.mode == :baseline_fmm
            N = size(X, 1)
            fname = @sprintf("baseline_%06d.csv", step)
            fpath = joinpath(cfg.save_dir, fname)
            _save_snapshot_csv(
                fpath,
                ["t", "dt", "step", "x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma",
                 "U_ref_x", "U_ref_y", "U_ref_z", "Gx_ref_x", "Gx_ref_y", "Gx_ref_z",
                 "Gy_ref_x", "Gy_ref_y", "Gy_ref_z", "Gz_ref_x", "Gz_ref_y", "Gz_ref_z"],
                fill(t, N), fill(dt, N), fill(step, N), X, Gamma, sigma, U_ref, Gx_ref, Gy_ref, Gz_ref,
            )
            if step == 0 || step % 10 == 0
                println("Task1 saved baseline snapshot: $(fpath)")
            end
        elseif cfg.mode == :surrogate_ml
            N = size(X, 1)
            U_ml, Gx_ml, Gy_ml, Gz_ml = get_particle_ugradu(pfield)
            fname = @sprintf("surrogate_%06d.csv", step)
            fpath = joinpath(cfg.save_dir, fname)
            _save_snapshot_csv(
                fpath,
                ["t", "dt", "step", "x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma",
                 "U_ml_x", "U_ml_y", "U_ml_z", "Gx_ml_x", "Gx_ml_y", "Gx_ml_z",
                 "Gy_ml_x", "Gy_ml_y", "Gy_ml_z", "Gz_ml_x", "Gz_ml_y", "Gz_ml_z"],
                fill(t, N), fill(dt, N), fill(step, N), X, Gamma, sigma, U_ml, Gx_ml, Gy_ml, Gz_ml,
            )
            if step == 0 || step % 10 == 0
                println("Task1 saved surrogate snapshot: $(fpath)")
            end
        else
            N = size(X, 1)
            U_ml, Gx_ml, Gy_ml, Gz_ml = get_particle_ugradu(pfield)
            dU = U_ml .- U_ref
            dGx = Gx_ml .- Gx_ref
            dGy = Gy_ml .- Gy_ref
            dGz = Gz_ml .- Gz_ref
            fname = @sprintf("compare_%06d.csv", step)
            fpath = joinpath(cfg.save_dir, fname)
            _save_snapshot_csv(
                fpath,
                ["t", "dt", "step", "x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma",
                 "U_ref_x", "U_ref_y", "U_ref_z", "Gx_ref_x", "Gx_ref_y", "Gx_ref_z",
                 "Gy_ref_x", "Gy_ref_y", "Gy_ref_z", "Gz_ref_x", "Gz_ref_y", "Gz_ref_z",
                 "U_ml_x", "U_ml_y", "U_ml_z", "Gx_ml_x", "Gx_ml_y", "Gx_ml_z",
                 "Gy_ml_x", "Gy_ml_y", "Gy_ml_z", "Gz_ml_x", "Gz_ml_y", "Gz_ml_z",
                 "dU_x", "dU_y", "dU_z", "dGx_x", "dGx_y", "dGx_z",
                 "dGy_x", "dGy_y", "dGy_z", "dGz_x", "dGz_y", "dGz_z"],
                fill(t, N), fill(dt, N), fill(step, N), X, Gamma, sigma,
                U_ref, Gx_ref, Gy_ref, Gz_ref, U_ml, Gx_ml, Gy_ml, Gz_ml, dU, dGx, dGy, dGz,
            )
            if step == 0 || step % 10 == 0
                println("Task1 saved compare snapshot: $(fpath)")
            end
        end

        return false
    end

    return cb
end

end # module
