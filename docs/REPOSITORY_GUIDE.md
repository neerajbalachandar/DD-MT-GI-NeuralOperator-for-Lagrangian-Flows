# Repository Guide and Review Notes

## Project Flow

`scripts/preprocess.py` calls `gino/data/preprocessing.py`, which configures and runs `gino/data/preprocessing_pipeline.py`. The pipeline reads Task-1 particle XMF/HDF5 and Task-2 Eulerian XMF/HDF5, matches field snapshots by XMF time, makes physical-identity particle transitions, derives train-only normalization/domain metadata, and writes `processed_data/particle_evolution_dataset.npz`.

`scripts/train.py` loads that archive, makes sequence splits and train-only normalization, builds `GINOSharedLatent`, and calls `gino/training/trainer.py`. The model shares an encoded GNO/FNO latent between particle-state and field decoders. `scripts/evaluate.py` checks the checkpoint contract, evaluates held-out one-step and autoregressive predictions, compares native Task-2 fields, writes CSV/JSON, and creates figures beneath the checkpoint's `evaluation/` directory.

## Review Findings and Interpretation

- Native Task-2 velocity is the stored HDF5 `U` at the XMF-declared nodal coordinates. The velocity evaluation now scores those three stored components directly, with no spatial interpolation or reconstruction of the truth. The twelve-channel model also predicts nine velocity-gradient channels; preprocessing derives these with physical-spacing finite differences. `field_all_channel_mse` is therefore an auxiliary score, not the primary direct-velocity result.
- The data uses Task-2 snapshots only at their actual times. Native evaluation additionally evaluates `field_superres_pair_ids`, where Task-2 targets were excluded from field training. These records have `evaluation_role=field_superresolution` in `native_task2_field.csv/json`; they test temporal interpolation only if those snapshots lie between training field times. They do not claim spatial super-resolution because queries use the native mesh.
- The evaluation report now produces one-step loss versus normalized simulation phase, rollout error versus step with cross-case mean/std, a native mid-slice `u_x` truth/prediction/error figure, `u_x/U_inf` versus `z/c` profiles, and true/predicted particle rollout snapshots.
- One-step particle and field errors are plotted against target physical time where available. Rollout plots show error at each generated target time and a separate mean-error-versus-horizon view. The configured training curriculum now reaches H=8. Validation includes autoregressive rollout loss when the validation sequences contain chains; a stride-held-out internal validation split may not contain adjacent transitions, so inspect `effective_horizon` before treating its rollout validation as meaningful.
- Train-only coordinate bounds are scientifically appropriate for leakage control. The model's latent-grid sampler clamps normalized query coordinates to [0,1], however. Evaluation records `outside_train_domain_fraction` for native meshes; nonzero values mean some test queries are clipped to the edge of the training domain and are extrapolation failures/limitations, not ordinary interpolation error.
- Particle rollout scores are only meaningful for verified persistent identities. The current loader rejects unverified correspondence for multistep rollouts; legacy row-index data must be regenerated or evaluated one-step-only.
- `training.use_pushforward` adds a separate detached-input objective. It is not equivalent to full differentiable BPTT. Geometry features recomputed through VTK/NumPy also form a stopped-gradient boundary. Avoid describing the complete geometry-conditioned trajectory as fully differentiable.
- The trainer currently uses autocast without gradient scaling, because GradScaler's CUDA unscale path failed on complex FNO gradients. With `mixed_precision: true`, monitor for non-finite or stalled losses; full precision is the conservative choice until a complex-gradient-safe scaling strategy is implemented.
- The architecture has an explicit `predict_delta_u` switch, but review the preprocessing target contract before changing it: the saved residual target has ten channels (seven state changes plus three velocity increments) when enabled. Do not treat those velocity increments as additional propagated state variables.
- The presentation plots now use direct native Task-2 `u_x` for the mid-slice comparison and downstream profiles. Profiles use `z/c`; if no chord is set explicitly, chord and leading-edge x are estimated from the VTK surface x bounds.
- Full raw-data preprocessing and a real-data training/evaluation run were not executed during this code review. Synthetic/unit tests cannot establish that all source files are complete or that every case metadata entry is physically correct.

## Tree Index

### Root and Configuration

| Path | Purpose |
|---|---|
| `README.md` | Short project introduction and current usage notes. |
| `requirements.txt` | Python dependencies for data processing, training, evaluation, and plotting. |
| `case_metadata.json` (when supplied) | Per-case physical metadata such as timestep, freestream, and test roles; the pipeline validates it. |
| `configs/default.yaml` | Default data paths, model dimensions/switches, training objectives/curriculum, and evaluation options. |

### Scripts

| Path | Purpose |
|---|---|
| `scripts/preprocess.py` | CLI for preprocessing; accepts Task-1 root, Task-2 root, and output directory. |
| `scripts/train.py` | Split selection, normalization, datasets/model construction, and training entry point. |
| `scripts/evaluate.py` | Checkpoint compatibility checks, one-step/rollout/native-field metrics, tables, and plots. |

### `gino/data`

| Path | Purpose |
|---|---|
| `gino/data/preprocessing.py` | Public wrapper around the pipeline; resolves source/output paths and preserves a legacy function alias. |
| `gino/data/preprocessing_pipeline.py` | Raw XMF/HDF5/VTK loading, time matching, particle correspondence, field validation, splits, feature construction, and archive writing. |
| `gino/data/preprocess_commented_uncommented.py` | Historical commented/alternate pipeline copy; not imported by the normal CLI. Treat the active `preprocessing_pipeline.py` as canonical. |
| `gino/data/hdf5.py` | Native Task-2 HDF5 readers, strict shape checks, physical-coordinate gradient calculation, and structured-plane extraction. |
| `gino/data/dataset.py` | Archive loading, sequence split utilities, particle/query subsampling, normalization, and rollout-chain construction. |
| `gino/data/context.py` | Immutable metadata carried between rollout transitions (frames, phase, time, geometry, and operating condition). |
| `gino/data/normalization.py` | Train-only input/target statistics and position normalization/denormalization. |
| `gino/data/geometry.py` | Particle geometry feature calculation used when rebuilding model inputs. |
| `gino/data/reconstruction.py` | Reconstructs the next particle input from predicted state/field, including geometry-conditioned features. |
| `gino/data/__init__.py` | Package marker and data API namespace. |

### `gino/model`

| Path | Purpose |
|---|---|
| `gino/model/gino.py` | Shared-latent GINO module, encoding, FNO processing, particle and field decoding. |
| `gino/model/gno.py` | Neural-operator-library GNO block construction. |
| `gino/model/fno.py` | FNO construction for the regular latent grid. |
| `gino/model/decoders.py` | MLP heads for particle transitions and field quantities. |
| `gino/model/attention.py` | Particle-query to shared-latent cross-attention. |
| `gino/model/positional_encoding.py` | Fourier coordinate encoding for particle and field queries. |
| `gino/model/adapters.py` | Task-specific transformations applied to shared latent features. |
| `gino/model/skip.py` | Residual/skip addition helper. |
| `gino/model/__init__.py` | Model package namespace. |

### `gino/dynamics`

| Path | Purpose |
|---|---|
| `gino/dynamics/physics.py` | VPM-inspired physical state transition/residual baseline. |
| `gino/dynamics/state_transition.py` | Combines model residual output with the physical transition and decodes field output. |
| `gino/dynamics/rollout.py` | Shared rollout engine for training, inference, and detached pushforward. |
| `gino/dynamics/__init__.py` | Dynamics package namespace. |

### `gino/training`

| Path | Purpose |
|---|---|
| `gino/training/trainer.py` | Epoch loops, gradient accumulation, rollout/pushforward objectives, sampling/noise, optimizer, checkpointing. |
| `gino/training/losses.py` | State, residual, field, rollout, and multitask weighting losses. |
| `gino/training/noise.py` | Temporally correlated GNS random-walk noise and flow-feature perturbation. |
| `gino/training/scheduled_sampling.py` | Sampling schedule, teacher-step indexing, and predicted/teacher flow selection. |
| `gino/training/pushforward.py` | Detached generated-input sequence helpers. |
| `gino/training/__init__.py` | Training package namespace. |

### `gino/evaluation`, `visualization`, and utilities

| Path | Purpose |
|---|---|
| `gino/evaluation/one_step.py` | One-step model inference and physical-unit metrics. |
| `gino/evaluation/autoregressive.py` | Reusable autoregressive evaluation helper. |
| `gino/evaluation/field.py` | Native field comparison helper. |
| `gino/evaluation/metrics.py` | MSE, relative L2, and per-state-component metric functions. |
| `gino/evaluation/__init__.py` | Evaluation package namespace. |
| `visualization/fields.py` | Native u_x truth/prediction/error panels and normalized downstream u_x profiles. |
| `visualization/temporal.py` | Log-scale one-step errors, rollout mean/std, and qualitative particle snapshots. |
| `visualization/evaluation.py` | State parity and coordinate-projected field error maps. |
| `visualization/__init__.py` | Visualization package namespace. |
| `gino/utils.py` | YAML config loading/overrides and JSON output helper. |
| `gino/__init__.py` | Top-level package namespace. |

### Tests, notebooks, and outputs

| Path | Purpose |
|---|---|
| `tests/test_model.py` | Model forward shapes, checkpoint roundtrip, shared encoding, architecture switches. |
| `tests/test_dataset.py` | Split leakage and train-only normalization behavior. |
| `tests/test_dataset_contract.py` | Identity, active-split rollout-chain, and field-target exclusion contracts. |
| `tests/test_particle_correspondence.py` | Identity matching, missing IDs, and duplicate-ID rejection. |
| `tests/test_preprocessing_integrity.py` | XMF/HDF5 layout, node ordering, scalar lengths, and rollout identity checks. |
| `tests/test_xmf_time.py` | XMF time parsing and physical timestep conversion. |
| `tests/test_hdf5_gradients.py` | Derivative correctness with nonuniform physical spacing and arbitrary HDF5 row ordering. |
| `tests/test_native_field_coordinates.py` | Native coordinate alignment safeguards for field comparisons. |
| `tests/test_reconstruction.py` | Rebuilt inputs and temporal context through rollout. |
| `tests/test_rollout.py` | Rollout propagation, terminal boundary, gradients, and pushforward behavior. |
| `tests/test_physics.py` | Physical transition equations. |
| `tests/test_training_mechanisms.py` | GNS correlation/indexing, teacher indexing, and loss normalization. |
| `notebooks/inspect_data.ipynb` | Interactive raw/processed data inspection. |
| `notebooks/inspect_training.ipynb` | Interactive training/checkpoint diagnostics. |
| `notebooks/analyze_results.ipynb` | Post-run metric and result analysis. |
| `notebooks/visualize_results.ipynb` | Interactive result plots. |
| `processed_data/` | Generated archive and merged intermediate frames; reproducible outputs, not source. |
| `runs/` | Training checkpoints, resolved configs, histories, and evaluation artifacts; preserve as experiment outputs. |
