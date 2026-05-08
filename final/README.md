Workflow:
1. Raw data directory -> merged NPZ frames
2. NPZ visual inspection
3. NPZ -> model ready datasets
4. Train models
5. Validate models
6. Super-resolution checks

---

## What each file does

- `h5_vtk_to_npz.py`: Convert split raw streams (`input`/`output`) into merged NPZ frames.
- `dataset_builder.py`: Build train-ready datasets from NPZ (field presets + particle dataset).
- `train.py`: Train field model (FNO) or particle model (GNOBlock).
- `sanity_checks.py`: Visual checks of data/model IO/training curves.
- `particle_rollout.py`: Quantity-model + physics rollout check for particle evolution.

Configs:
- `configs/h5_vtk_to_npz_template.yaml`
- `configs/pipeline_config.yaml`
- `configs/train_field_fno.yaml`
- `configs/train_particle_gno.yaml`
- `configs/particle_rollout.yaml`

---

### 1A. Fill raw input/output paths

Edit `configs/h5_vtk_to_npz_template.yaml`.

For each dataset entry, set:
- `input_h5_glob`
- `output_h5_glob`
- optional: `input_xmf_glob`, `output_xmf_glob`, `vtk_glob`
- `input_key_map` and `output_key_map`

### 1B. Run conversion

```bash
python -m final.h5_vtk_to_npz \
  --config final/configs/h5_vtk_to_npz_template.yaml \
  --root "/path/to/raw_data_parent"
```

### 1C. Check conversion result

Check:
- `final/output/merged_npz/<dataset_id>/conversion_summary.json`
- `final/output/merged_npz/<dataset_id>/conversion_index.json`

If `n_written = 0`, fix globs/key maps first.

---

## Task 2: Visual sanity on merged NPZ (before training)

```bash
python -m final.sanity_checks --skip-model-forward
```

Outputs:
- `final/output/sanity_checks/merged_inputs/*`

This lets you inspect:
- detected files
- particle cloud bounds
- sample scatter plots

---

## Task 3: NPZ -> model-ready datasets

### 3A. Fill NPZ dataset config

Edit `configs/pipeline_config.yaml`:
- `cases[].npz_glob` must point to merged NPZ from Task 1
- set `grid.resolution` (for example `[32, 32, 32]`)
- choose presets and `output_mode`

### 3B. Build datasets

```bash
python -m final.dataset_builder \
  --config final/configs/pipeline_config.yaml \
  --build-field --build-particle
```

Generated folders (example):
- `final/output/unified_vpm_geometry_v2/preset_E/`
- `final/output/unified_vpm_geometry_v2/particle_dataset/`

---

## Task 4: Train models

### 4A. Field model (FNO)

Edit `final/configs/train_field_fno.yaml`:
- `preset_root`
- `output_dir`
- `device` (`cuda` recommended on GPU machine)

Run:
```bash
python -m final.train --task field --config final/configs/train_field_fno.yaml
```

### 4B. Particle model (GNOBlock)

Edit `final/configs/train_particle_gno.yaml`:
- `dataset_npz`
- `split_idx_npz`
- `output_dir`
- `device`

Run:
```bash
python -m final.train --task particle --config final/configs/train_particle_gno.yaml
```

---

## Task 5: Validation and model behavior checks

After training, run full sanity:

```bash
python -m final.sanity_checks \
  --pipeline-config final/configs/pipeline_config.yaml \
  --field-train-config final/configs/train_field_fno.yaml \
  --particle-train-config final/configs/train_particle_gno.yaml
```

This produces:
- input/target slices
- prediction/error slices (if checkpoints exist)
- training history plots

Particle surrogate rollout check:

```bash
python -m final.particle_rollout --config final/configs/particle_rollout.yaml
```

This reports:
- quantity-level error (`U`, `gradU`)
- one-step rollout error (`X`, `Gamma`, `sigma`)

---

## Task 6: Super-resolution checks (manual setup)

Use a separate SR dataset path/config (recommended), not the training set.

### 6A. Convert SR raw files to merged NPZ
- duplicate `h5_vtk_to_npz_template.yaml` for SR and point to SR raw folders
- choose separate `output_subdir` values (for example `sr_test`)

### 6B. Build SR field dataset
- duplicate `pipeline_config.yaml` for SR
- set `cases[].npz_glob` to SR merged NPZ
- set higher resolution, e.g. `grid.resolution: [64, 64, 64]`
- build with `final.dataset_builder`

### 6C. Run SR inference sanity
- in `train_field_fno.yaml`:
  - set `preset_root` to SR preset folder
  - keep `output_dir` pointing to trained checkpoint folder
- run `final.sanity_checks` (without `--skip-model-forward`)

This gives prediction-vs-target plots on SR grids using the trained model.

---

## Important modeling note (particle surrogate)

Current implementation is **quantities_then_physics**:
- learn local interaction quantities (`velocity`, `velocity_gradient`)
- then advance particles with physics update (rVPM-style)

This is preferred over pure black-box next-state learning when particle correspondence across frames is unstable.

---

## Quick command order summary

```bash
# 1) Raw -> merged NPZ
python -m final.h5_vtk_to_npz --config final/configs/h5_vtk_to_npz_template.yaml --root "/path/to/raw"

# 2) Pre-build sanity
python -m final.sanity_checks --skip-model-forward

# 3) Build model-ready datasets
python -m final.dataset_builder --config final/configs/pipeline_config.yaml --build-field --build-particle

# 4) Train
python -m final.train --task field --config final/configs/train_field_fno.yaml
python -m final.train --task particle --config final/configs/train_particle_gno.yaml

# 5) Validate + rollout
python -m final.sanity_checks --pipeline-config final/configs/pipeline_config.yaml --field-train-config final/configs/train_field_fno.yaml --particle-train-config final/configs/train_particle_gno.yaml
python -m final.particle_rollout --config final/configs/particle_rollout.yaml
```
