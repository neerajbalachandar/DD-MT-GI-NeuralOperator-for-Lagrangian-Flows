# VortexGINO: what was wrong, what changed, and where the model stands

Final model: `runs/vortex_gino/final_model.pt` (EMA weights of joint-phase epoch 30 of
`configs/vortex_gino.yaml`, rolled out with FLOWVPM corrected-Pedrizzetti relaxation).
Evaluation: `runs/vortex_gino/evaluation_final/` (`summary.md`, `summary.csv`, `metrics.json`, `plots/`).
Reproduce: `python scripts/build_vortex_dataset.py && python scripts/train_vortex.py && python scripts/evaluate_vortex.py --checkpoint runs/vortex_gino/final_model.pt`.

## 1. Why the old pipeline could not learn

Each item below was verified numerically (audit scripts and an adversarial verification pass).

| # | Defect | Effect |
|---|---|---|
| 1 | N320/N640 cases truncated to 2048 particles (`MAX_PREPROCESS_PARTICLES=2048`) | inputs carried 71% / 36% of the circulation that produced the field targets; Biot-Savart of the stored particles missed 29% / 65% of the velocity |
| 2 | Field decoder sampled the latent with x and z swapped (`grid_sample` axis order) | encoder writes at (x,y,z), decoder reads at (z,y,x) |
| 3 | Absolute, anisotropic coordinates: 16^3 latent over 5 x 5 x 17 (spacing 0.33 x 0.33 x 1.13), FNO kept only k <= 3, mean-reduction GNO | rings (sigma = 0.1, separation 0.75-2) occupy 1-4 latent cells in z; mean reduction breaks superposition and particle-resolution invariance |
| 4 | Explicit-Euler physics base over a saved interval of 4 solver steps | explains only ~40-45% of the displacement (core particles rotate ~1.5 rad per frame); residual head had to learn the rest |
| 5 | Classic stretching used, FLOWVPM runs `transposed=true` | wrong physics base |
| 6 | Exact particle velocity/gradient present in the data but unused | particle-located velocity supervised only indirectly |
| 7 | Frame 0 has all-zero FLOWVPM particle u/J | first pair of every case had a corrupted residual target |
| 8 | State/rollout loss normalised by absolute state std; residual and state targets in conflict; homoscedastic weights + plateau LR (LR decayed to 5e-9); phase conditioning | position errors invisible, Gamma_z noise dominated, training stalled |
| 9 | `scripts/evaluate.py` crashed (`KeyError: 'plane_specs'`) | no evaluation plots or rollout metrics were ever written |
| 10 | Finite-difference gradient targets on a 0.16 x 0.16 x 0.51 grid vs sigma = 0.1 cores | aliased targets (FD curl vs analytic W: corr. 0.7, ~0 for omega_z) |

Old results (validation): field-velocity normalised MSE 0.6-0.9, vorticity ~0.97, residual ~0.85 (barely better than predicting the mean).

## 2. What changed

**Data** (`gino/data/vortex_raw.py`, `scripts/build_vortex_dataset.py`): lossless per-case export straight from the raw FLOWVPM HDF5 (19 cases: 8 train / 4 val / 7 test; the old archive had dropped 4, including the isolated-ring training case), all particles, exact particle u and J (J completed with the self-induced term FLOWVPM omits; now matches the exact Biot-Savart gradient to 0.03%), frame-0 masking, grid U replaced by the exact Biot-Savart field of the particles (FLOWVPM's grid U includes injected probe particles: 1.4-5% off, 23% on 64^3), analytic grid vorticity. Builds in ~8 min instead of the old multi-stage pipeline.

**Model** (`gino/model/vortex_operator.py`, `VortexGINO`): one shared operator particles -> (u, J) at any point, used for both tasks.
- co-moving isotropic latent box (24 x 24 x 28, spacing ~0.24) around the circulation centroid;
- learned particle-to-grid encoder, linear in Gamma and summed (superposition; N320 and N640 latents are identical);
- neuralop FNO (modes 12/12/14, domain padding for the non-periodic box);
- decoder = latent sample + rotation-equivariant learned near-field Biot-Savart-type kernel with analytic gradient (resolves sigma = 0.1 cores below the latent spacing);
- dynamics: dx/dt = u(x); dGamma/dt = |Gamma| r_theta, a learned relative circulation rate supervised with the exact `J^T Gamma/|Gamma|` (the stretching term is a ~3% residue of |J||Gamma|, so using a learned J directly was exponentially unstable); 2 coupled RK4 substeps per saved interval (1 step collapses the cores even with exact physics); FLOWVPM corrected-Pedrizzetti relaxation with the model's own vorticity (keeps Gamma aligned with omega, which removed the circulation blow-up); learned residual.

**Training** (`gino/training/vortex_trainer.py`): relative-L2 losses on grid U, grid W, particle u, particle J, circulation rate; operator phase, then joint phase with multi-step coupled-RK4 rollout losses (checkpointed BPTT, deterministic scatter for reproducible recomputation); importance-sampled grid queries; warmup + cosine LR; EMA weights (the raw Adam iterate oscillated); per-phase best checkpoints; resume with optimizer state. Optional stage 3 (`configs/vortex_gino_stage3.yaml`): on-policy training on states visited by the model's own rollouts, with exact Biot-Savart targets.

**Evaluation** (`scripts/evaluate_vortex.py`, `visualization/vortex_plots.py`): teacher-forced field reconstruction (velocity, vorticity, vortex-core vs far-field shares, particle u/J/circulation rate), 180-step rollouts with persistence baseline, ring centroid z(t) and radius R(t), circulation and impulse conservation, field decoded from rolled-out particles, divergence detection that does not use the truth, `summary.md` with plain-language metric definitions.

**Legacy fixes**: x/z swap in `gino/model/gino.py`, transposed stretching in `gino/dynamics/physics.py`, `evaluate.py` crash, particle cap in preprocessing and configs.

**Tests**: `tests/test_vortex_operator.py`, `tests/test_vortex_data.py` (analytic gradient, rotation equivariance, superposition, translation invariance, latent axis order, RK4 exactness and substep convergence, transposed scheme, relaxation vs the FLOWVPM formula, data vs exact Biot-Savart).

## 3. Results (final model, `runs/vortex_gino/evaluation_final/summary.md`)

Field reconstruction from the true particles (relative L2, mean over frames):

| group | grid velocity | grid vorticity | particle velocity | particle grad u | circulation rate |
|---|---|---|---|---|---|
| validation (4 cases, incl. unseen N640) | 0.063 | 0.115 | 0.036 | 0.052 | 0.076 |
| in-regime test (2 cases) | 0.052 | 0.045 | 0.029 | 0.039 | 0.086 |
| 64^3 grid (seen IC; resolution check) | 0.050 | 0.044 | 0.029 | - | 0.077 |

Autoregressive rollouts (180 steps from frame 0):
- no in-regime or validation rollout diverges; at step 180 total circulation is within 2-11% of the truth (18% for the unseen N640 resolution); before the fixes every rollout blew up by step ~60-80;
- mean particle position error / ring radius: 0.10-0.17 at step 45, 0.16-0.39 at step 90, 1.0-2.1 at step 180 (frozen-particle baseline 6.1-10.5);
- the first leapfrog cycle is reproduced on unseen cases (rear ring shrinks to R ~0.55, passes through, front ring grows to ~1.3; radii track the truth for ~95 steps and centroids for ~125 steps); the second cycle is not;
- velocity field decoded from the rolled-out particles: 0.13-0.20 relL2 at step 30 (0.59 for the widely spaced d = 2.0 val case), 0.23-0.82 at step 60, ~1 by step 120 (rings displaced);
- 4x smaller time step (temporal-SR perturbed ring, not trained on): 0.17 R0 after 180 steps (baseline 1.73).

Out of regime: the free pair (rings 5 apart) leaves the co-moving box and fails; the toroidal ring (sigma 0.15, Gamma 20) reconstructs at 0.26 velocity error and diverges at step ~105; the d = 2.5, Gamma-ratio 0.25 pair is stable but inaccurate (0.12 field error).

## 4. Known limitations and next steps
- Long-horizon accuracy is limited by the operator's ~3-5% particle-velocity error compounding through the leapfrog pass-through; the second leapfrog cycle is not captured. Options: larger/finer latent box, more training cases (only 8 trajectories), longer BPTT windows with pushforward, on-policy training for longer (stage 3 did not improve the 100-step validation rollout in 6 epochs).
- The co-moving box (+/-2.8 x 2.8 x 3.3) cannot represent widely separated vortex systems (free pair); a multi-box or adaptive-extent latent would be needed.
- Only vortex-ring data was used; the roundjet cases require the same export plus inflow handling.
