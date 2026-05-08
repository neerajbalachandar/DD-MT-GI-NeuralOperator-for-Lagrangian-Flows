# final-2 (Simple Manual Workflow)

You asked for a non-automated, easy-to-debug version.
This folder has only:
- `preprocess_data.py` (single preprocessing script)
- `notebooks/task1_particle_field_evolution.ipynb`
- `notebooks/task2_velocity_field_reconstruction.ipynb`

## Step 1: preprocess
Edit user settings in `preprocess_data.py`, then run:

```bash
python3 final-2/preprocess_data.py
```

Creates:
- `final-2/output/field_dataset.npz`
- `final-2/output/particle_dataset.npz`
- `final-2/output/merged_frames/*`

## Step 2: run notebooks
Run Task-1 notebook for particle surrogate (GNO), Task-2 notebook for field reconstruction (FNO).

Both notebooks automatically use CUDA if available.


## Path behavior
- `preprocess_data.py` always writes to `final-2/output` relative to its own file location.
- Notebooks auto-detect the `final-2` base path so they work whether launched from repo root or from `final-2/notebooks`.
