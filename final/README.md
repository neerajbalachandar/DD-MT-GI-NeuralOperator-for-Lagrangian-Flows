# Final Geometry-Aware Pipeline (New Files Only)

This folder is a **new version** of the preprocessing/training pipeline. Existing files are untouched.

## What is included

- `io.py` - config loading, file discovery, split/index helpers
- `vtk_loader.py` - generic VTK parser (legacy + XML, point/cell arrays auto-detected)
- `projection.py` - modular particle-to-grid projection (Gaussian/compact kernels)
- `geometry_features.py` - body mask, SDF, normals, surface velocity, global channels
- `interpolation.py` - remapping hierarchy: direct -> trilinear -> griddata -> RBF
- `normalization.py` - train-split dataset-level channel normalization stats
- `dataset_builder.py` - builds presets A-F + particle-surrogate dataset
- `train.py` - field FNO training + particle true GNOBlock model
- `h5_vtk_to_npz.py` - merges split input/output H5 streams (+ optional XMF/VTK) into unified NPZ
- `sanity_checks.py` - visual sanity pass (data summaries, channel slices, model forward checks, training curves)
- `particle_rollout.py` - one-step particle rollout check using learned quantities (`U`, `gradU`) + rVPM-style state update

## Key support implemented

- Full Gamma vector kept (`Gamma_x`, `Gamma_y`, `Gamma_z`) and projected independently.
- Configurable input channels, output modes (`U`, `W`, `UW`), geometry conditioning.
- Shared pipeline for stationary and flapping/moving cases.
- Particle-level dataset (`inputs_particle`, `targets_particle`) for surrogate training.
- Logging outputs for channels, interpolation mode, missing files, VTK arrays.

## Geometry Logic (Generalized)

- Geometry channels are sampled from either point arrays or cell arrays (via cell centers), then mapped to the target grid by nearest-neighbor.
- `body_mask` and `signed_distance_field` are computed from surface points (with signed convention when normals are available).
- Surface normals are taken from available arrays or reconstructed from mesh normals.
- Surface velocity uses:
  - explicit VTK arrays first (e.g. `Vkin`)
  - mesh-motion finite difference fallback
  - zero fallback
- For stationary cases (`stationary: true`), `phase=0` and surface-velocity channels are forced to zero.
- Freestream channels come from case metadata when provided, otherwise inferred from VTK freestream-like arrays when available.

## Sample VTK Mapping (Flapping Wing)

From sample file `p_p_flapping_Wing_L_vlm.1478.vtk`, detected cell arrays include:
- `Vinf` -> `freestream_x/y/z`
- `Gamma` -> `panel_gamma`
- `Vkin` -> `surface_velocity_x/y/z`
- `Vvpm`, `Vvpm_AB`, `Vvpm_ApA`, `Vvpm_BBp` -> induced surface velocity channels
- `L` -> `lift_x/y/z`
- `D` -> `drag_x/y/z`
- `S` -> `sideforce_x/y/z`
- `Ftot` / `ftot` -> `total_force_x/y/z`

These aliases are implemented in `final/geometry_features.py`.

## Two-Stage Workflow (Preprocess and Train Separated)

1. Convert raw H5/XMF/VTK to merged NPZ frames (CPU/preprocess machine):

```bash
python -m final.h5_vtk_to_npz --config final/configs/h5_vtk_to_npz_template.yaml
```

2. Build train-ready datasets from merged NPZ (still preprocess machine):

```bash
python -m final.dataset_builder --config final/configs/pipeline_config.yaml --build-field --build-particle
```

3. Transfer generated dataset folders (for example `final/output/unified_vpm_geometry_v2`) to GPU machine via USB.

4. Train on GPU machine (CUDA enabled via `device: cuda` in train config):

```bash
python -m final.train --task field --config final/configs/train_field_fno.yaml
python -m final.train --task particle --config final/configs/train_particle_gno.yaml
```

5. Run sanity checks (optional but strongly recommended):

```bash
python -m final.sanity_checks \
  --pipeline-config final/configs/pipeline_config.yaml \
  --field-train-config final/configs/train_field_fno.yaml \
  --particle-train-config final/configs/train_particle_gno.yaml
```

6. Evaluate particle rollout (after particle model training):

```bash
python -m final.particle_rollout --config final/configs/particle_rollout.yaml
```

This reports two levels of quality:
- quantity-level: how well model predicts `velocity` and `velocity_gradient`
- rollout-level: how well one-step updated `X/Gamma/sigma` match the next frame

Artifacts are written under `final/output/sanity_checks/`:
- merged-input summaries and scatter plots
- sample input/target channel slices
- (if checkpoints exist) prediction vs target/error plots
- training-history curves from `history.json`

Use separate dataset fields in `h5_vtk_to_npz_template.yaml`:
`input_h5_glob`, `output_h5_glob`, `input_xmf_glob`, `output_xmf_glob`, `vtk_glob`.

## Notes

- `final/configs/pipeline_config.yaml` is stage-2 and consumes NPZs from `h5_vtk_to_npz.py`.
- Cases are fully generic (e.g. `1`, `2`, `7`, `8`); no flapping/stationary classification is required.
- `h5_vtk_to_npz.py` is simplified to the split input/output workflow (no legacy single-stream branch).
- particle surrogate follows `quantities_then_physics`: learn local P2P quantities first, then advance state with physics update.
- VTK loading uses `pyvista` when available, then falls back to `vtk`.
- Interpolation methods requiring SciPy will gracefully fall back if unavailable.
- Optimizer selection is `auto` by default and chooses Adam/AdamW from NeuralOperator when available.

# File order

- Config YAML
- h5_vtk_to_npz.py
- dataset_builder.py
- inspect data
- train.py --task field
- train.py --task particle
- ablations
- paper figures
