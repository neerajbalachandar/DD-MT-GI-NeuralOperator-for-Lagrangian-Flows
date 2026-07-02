module Task1FlowUnsteadyRuntime

using DelimitedFiles
using Printf

include("task1_particle_io.jl")
include("task1_python_predictor.jl")

using .Task1ParticleIO
using .Task1PythonPredictor

export build_runtime_callback, build_ml_uj_function

function _save_snapshot_csv(fpath::String, columns::Vector{String}, arrays...)
    data = hcat(arrays...)
    open(fpath, "w") do io
        println(io, join(columns, ","))
        writedlm(io, data, ',')
    end
end

function _predict_ugradu!(pfield, cfg, predictor)
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
    Y = predictor(feats)
    set_particle_ugradu!(pfield, Y)
    return true
end

function build_ml_uj_function(cfg)
    predictor, close_predictor = build_predictor(cfg.model_py_path, cfg.model_meta_npz; device=cfg.device)
    atexit(close_predictor)

    total_calls = Ref(0)
    total_predictions = Ref(0)
    total_seconds = Ref(0.0)
    last_step = Ref(-1)
    step_calls = Ref(0)
    step_predictions = Ref(0)
    step_seconds = Ref(0.0)
    step_particles = Ref(0)
    predicted_step = Ref(-1)

    function flush_step_stats(force::Bool=false)
        if last_step[] < 0 || step_calls[] == 0
            return
        end
        if force || last_step[] == 1 || last_step[] % 10 == 0
            @printf(
                "Task1 ML UJ stats: step=%d calls_this_step=%d predictions_this_step=%d particles=%d ml_seconds=%.3f total_calls=%d total_predictions=%d total_ml_seconds=%.3f cache_per_step=%s\n",
                last_step[],
                step_calls[],
                step_predictions[],
                step_particles[],
                step_seconds[],
                total_calls[],
                total_predictions[],
                total_seconds[],
                string(cfg.cache_ml_uj_per_step),
            )
            flush(stdout)
        end
    end

    function ml_uj(pfield; optargs...)
        step = Int(getfield(pfield, :nt))
        if last_step[] != step
            flush_step_stats(false)
            last_step[] = step
            step_calls[] = 0
            step_predictions[] = 0
            step_seconds[] = 0.0
            step_particles[] = 0
        end

        total_calls[] += 1
        step_calls[] += 1
        step_particles[] = max(step_particles[], Int(getfield(pfield, :np)))

        if cfg.cache_ml_uj_per_step && predicted_step[] == step
            return nothing
        end

        elapsed = @elapsed _predict_ugradu!(pfield, cfg, predictor)
        predicted_step[] = step
        total_predictions[] += 1
        total_seconds[] += elapsed
        step_predictions[] += 1
        step_seconds[] += elapsed
        return nothing
    end

    atexit(() -> flush_step_stats(true))

    return ml_uj, predictor, close_predictor
end

"""
Build runtime callback to:
- export baseline U/gradU supervision
- or replace U/gradU with ML predictions
- or compare both in one run
"""
function build_runtime_callback(cfg; predictor=nothing, predict_in_callback::Bool=true)
    mkpath(cfg.save_dir)
    println("Task1 runtime callback active: mode=$(cfg.mode), save_dir=$(cfg.save_dir)")

    if predictor === nothing && predict_in_callback && (cfg.mode == :surrogate_ml || cfg.mode == :compare)
        predictor, close_predictor = build_predictor(cfg.model_py_path, cfg.model_meta_npz; device=cfg.device)
        atexit(close_predictor)
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

        # Baseline references before overwrite
        U_ref, Gx_ref, Gy_ref, Gz_ref = get_particle_ugradu(pfield)

        if predict_in_callback && (cfg.mode == :surrogate_ml || cfg.mode == :compare)
            _predict_ugradu!(pfield, cfg, predictor)
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
