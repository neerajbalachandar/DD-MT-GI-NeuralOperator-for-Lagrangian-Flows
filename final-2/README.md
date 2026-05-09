# final-2 (Task-1 First, Manual Workflow)

This folder is now focused on **Task-1 particle evolution**.

## Files
- `preprocess_data.py`
- `notebooks/task1_particle_field_evolution.ipynb`
- `notebooks/task2_velocity_field_reconstruction.ipynb` (left as-is for later)

## Task-1 Workflow
1. Edit metadata and split in `preprocess_data.py`:
   - `CASE_METADATA` (optional unless you add metadata channels back to input features)
   - `TRAIN_CASES`, `VAL_CASES`, `TEST_CASES` (case-level split)
2. Run preprocessing:
   - `python3 final-2/preprocess_data.py`
3. Open and run Task-1 notebook:
   - `final-2/notebooks/task1_particle_field_evolution.ipynb`

## Output files (Task-1)
- `final-2/output/particle_evolution_dataset.npz`
- `final-2/output/task1_evolution_training/best_particle_evolution_gno.pt`
- `final-2/output/task1_evolution_training/history.json`
- `final-2/output/task1_evolution_training/rollout_predictions.npz`
- `final-2/output/task1_evolution_training/rollout_error_arrays.npz`
- `final-2/output/task1_evolution_training/rollout_frames/*.png`
- `final-2/output/task1_evolution_training/rollout_animation.gif` (if imageio available)

## Notes
- Task-1 now learns: `x_t -> Delta x_t` (not instantaneous `x_t -> U,gradU`).
- Rollout validation is autoregressive and starts from frame 0 only.
- Normalization uses training split rows only.
