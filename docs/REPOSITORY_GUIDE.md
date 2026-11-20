# Repository Guide and Review Notes

## Current pipeline (VortexGINO, November 2026)

| Path | Purpose |
|---|---|
| `gino/data/vortex_raw.py`, `scripts/build_vortex_dataset.py` | Lossless per-case export of raw FLOWVPM output to `processed_data/vortex_rings_v2/<case>.npz` (+ `manifest.json`): all particles, exact particle velocity/gradient (self term completed), exact Winckelmans Biot-Savart grid velocity, FLOWVPM vorticity. Also the exact analytic evaluator `winckelmans_velocity_jacobian`. |
| `gino/data/vortex_frames.py` | GPU-resident case store, ring membership/diagnostics. |
| `gino/model/vortex_operator.py` | `VortexGINO`: co-moving latent box, Gamma-linear summed P2G encoder, FNO, decoder with equivariant near-field kernel (analytic gradient), learned circulation-rate head, coupled RK4 substep dynamics + learned residual. |
| `gino/training/vortex_trainer.py`, `scripts/train_vortex.py`, `configs/vortex_gino.yaml` | Operator phase (grid U/W, particle u/J, circulation rate) then joint phase with multi-step RK4 rollout losses; EMA weights; per-phase best checkpoints; resume with optimizer state; optional off-trajectory augmentation with exact Biot-Savart targets. |
| `scripts/evaluate_vortex.py`, `visualization/vortex_plots.py` | Teacher-forced field reconstruction + 180-step rollouts, conservation diagnostics, `summary.md`, clear per-case plots. |
| `tests/test_vortex_operator.py`, `tests/test_vortex_data.py` | Physics/invariance/data-consistency tests. |

## Defects found in the legacy pipeline (and how the new pipeline addresses them)

Data
- N320/N640 cases were truncated to 2048 of 2880/5760 particles (`MAX_PREPROCESS_PARTICLES=2048`), keeping only 71%/36% of the circulation that produced the field targets. -> all particles kept; legacy cap raised.
- Frame 0 of every case has zero FLOWVPM particle velocity/gradient (written before the first evaluation) and was used as a target. -> masked.
- FLOWVPM particle J omits the self-induced term, so it is not a sample of the continuous field the decoder predicts (irreducible ~15-22% error). -> completed with `2.5/(4 pi sigma^3)[Gamma]x`.
- FLOWVPM grid U includes the effect of ~32k injected grid probe particles (1.4-5% off, 23% on the 64^3 grid). -> replaced with the exact Biot-Savart field of the particles.
- Field gradients were finite differences on a 0.16 x 0.16 x 0.51 grid against sigma = 0.1 cores (aliased). -> analytic vorticity W is used instead.

Model
- The field decoder sampled the latent with x and z swapped (grid_sample axis order). Fixed in `gino/model/gino.py` (+ regression test).
- Anisotropic absolute-frame normalisation: 16^3 latent spacing 0.33 x 0.33 x 1.13 over a 5 x 5 x 17 box; FNO kept only |k|<=3; mean-reduction GNO (breaks Biot-Savart superposition and resolution invariance). -> co-moving isotropic box, summed Gamma-linear encoder, more modes, domain padding.
- Explicit Euler over a 4-solver-step interval explained only ~40-45% of the displacement (core particles rotate ~1.5 rad per saved frame); even exact physics with 1 RK4 step collapses the cores. -> 2 coupled RK4 substeps (exact physics: 0.3% one-step error).
- Stretching used the classic scheme while FLOWVPM runs transposed. Fixed in `gino/dynamics/physics.py`. In the new model, J^T Gamma is a ~3% residue of |J||Gamma|, so a learned J makes it exponentially unstable -> learned relative circulation rate supervised with the exact J^T Gamma/|Gamma|.

Training / evaluation
- State/rollout losses normalised by the absolute state std (position errors vanished, Gamma_z noise dominated); conflicting residual vs state targets; homoscedastic weights + plateau LR collapsed the LR to 5e-9; phase/freestream conditioning invited memorisation. -> relative losses (grid/particle fields; one-step errors relative to the true displacement), fixed weights, warmup+cosine LR, EMA weights, no global conditioning.
- `scripts/evaluate.py` crashed with `KeyError: 'plane_specs'`, so no plots/metrics were written. Fixed; plotting failures are now non-fatal.
- The spatial-SR test case has the same initial condition as a training case; it is reported only as a field super-resolution check.


## Project Flow

`scripts/preprocess.py` calls `gino/data/preprocessing.py`, which configures and runs `gino/data/preprocessing_pipeline.py`. The pipeline reads Task-1 particle XMF/HDF5 and Task-2 Eulerian XMF/HDF5, matches field snapshots by XMF time, makes physical-identity particle transitions, derives train-only normalization/domain metadata, and writes `processed_data/particle_evolution_dataset.npz`.

`scripts/train.py` loads that archive, makes sequence splits and train-only normalization, builds `GINOSharedLatent`, and calls `gino/training/trainer.py`. The model shares an encoded GNO/FNO latent between particle-state and field decoders. `scripts/evaluate.py` checks the checkpoint contract, evaluates held-out one-step and autoregressive predictions, compares native Task-2 fields, writes CSV/JSON, and creates figures beneath the checkpoint's `evaluation/` directory.

## Review Findings and Interpretation

- Native Task-2 velocity is the stored HDF5 `U` at the XMF-declared nodal coordinates. The velocity evaluation now scores those three stored components directly, with no spatial interpolation or reconstruction of the truth. The twelve-channel model also predicts nine velocity-gradient channels; preprocessing derives these with physical-spacing finite differences. `field_all_channel_mse` is therefore an auxiliary score, not the primary direct-velocity result.
- The data uses Task-2 snapshots only at their actual times. Native evaluation additionally evaluates `field_superres_pair_ids`, where Task-2 targets were excluded from field training. These records have `evaluation_role=field_superresolution` in `native_task2_field.csv/json`; they test temporal interpolation only if those snapshots lie between training field times. They do not claim spatial super-resolution because queries use the native mesh.
- The selected vortex-ring data uses a fixed particle population and stable file row ordering. Preprocessing uses canonical row identities when every frame in a case has the same particle count; variable-count cases keep the existing identity/track matching path.
- Vortex-ring generation defaults to a 32×32×32 Eulerian grid and writes particle and field XMF/HDF5 snapshots every four simulation steps. Preprocessing and the training config retain all 32,768 field nodes per snapshot. The temporal holdout can reserve field labels between training field times for later temporal super-resolution evaluation.
- Solid-body geometry conditioning is disabled for the selected free-flow data. Particle distance, surface-normal, and near-body channels are excluded from the active feature list; evaluation skips chord-normalized profiles when no reference chord or solid geometry exists.
- The evaluation report now produces one-step loss versus normalized simulation phase, rollout error versus step with cross-case mean/std, a native mid-slice `u_x` truth/prediction/error figure, `u_x/U_inf` versus `z/c` profiles, and true/predicted particle rollout snapshots.
- One-step particle and field errors are plotted against target physical time where available. Rollout plots show error at each generated target time and a separate mean-error-versus-horizon view. The configured training curriculum now reaches H=8. Validation includes autoregressive rollout loss when the validation sequences contain chains; a stride-held-out internal validation split may not contain adjacent transitions, so inspect `effective_horizon` before treating its rollout validation as meaningful.
- Train-only coordinate bounds are scientifically appropriate for leakage control. The model's latent-grid sampler clamps normalized query coordinates to [0,1], however. Evaluation records `outside_train_domain_fraction` for native meshes; nonzero values mean some test queries are clipped to the edge of the training domain and are extrapolation failures/limitations, not ordinary interpolation error.
- Particle rollout scores are only meaningful for verified persistent identities. The current loader rejects unverified correspondence for multistep rollouts; legacy row-index data must be regenerated or evaluated one-step-only.
- `training.use_pushforward` adds a separate detached-input objective. It is not equivalent to full differentiable BPTT.
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
| `gino/data/context.py` | Immutable metadata carried between rollout transitions (frames, phase, time, and operating condition). |
| `gino/data/normalization.py` | Train-only input/target statistics and position normalization/denormalization. |
| `gino/data/geometry.py` | Legacy solid-geometry helpers; not used for the active vortex-ring input channels. |
| `gino/data/reconstruction.py` | Reconstructs the next particle input from predicted state and field. |
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
