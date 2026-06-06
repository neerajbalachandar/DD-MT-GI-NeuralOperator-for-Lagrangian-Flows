# final-2 (Task-1 First, Manual Workflow)

This folder is now focused on **Task-1 particle evolution**.

## Files
- `preprocess_data.py`
- `notebooks/task1_particle_field_evolution.ipynb`
- `notebooks/task1_particle_field_evolution_2.ipynb` (x_t -> [u_t, gradU_t] mode)
- `notebooks/task2_velocity_field_reconstruction.ipynb` (left as-is for later)
- `src/` (Julia helpers for FLOWUnsteady integration without package edits)

## Task-1 Workflow
1. Edit metadata and split in `preprocess_data.py`:
   - `CASE_METADATA` (optional unless you add metadata channels back to input features)
   - `TRAIN_CASES`, `VAL_CASES`, `TEST_CASES` (case-level split)
2. Choose Task-1 mode in `preprocess_data.py`:
   - `TASK1_TARGET_MODE = "delta"` for `x_t -> Delta x_t`
   - `TASK1_TARGET_MODE = "ugradu"` for `x_t -> [u_t, gradU_t]`
3. Run preprocessing:
   - `python3 final-2/preprocess_data.py`
4. Open and run Task-1 notebook:
   - `final-2/notebooks/task1_particle_field_evolution.ipynb`
   - or `final-2/notebooks/task1_particle_field_evolution_2.ipynb` for u/gradU mode

## Output files (Task-1)
- `final-2/output/particle_evolution_dataset.npz`
- `final-2/output/task1_evolution_training/best_particle_evolution_gno.pt`
- `final-2/output/task1_evolution_training/history.json`
- `final-2/output/task1_evolution_training/rollout_predictions.npz`
- `final-2/output/task1_evolution_training/rollout_error_arrays.npz`
- `final-2/output/task1_evolution_training/rollout_frames/*.png`
- `final-2/output/task1_evolution_training/rollout_animation.gif` (if imageio available)

## Task-1 Input Channels (u/gradU mode)
Current default input feature vector:
- `x, y, z, Gamma_x, Gamma_y, Gamma_z, sigma`

Optional channels can be added later:
- `phase, angle_of_attack, freestream_x, freestream_y, freestream_z`
- geometry (now supported in preprocessing):
  - `geom_dist, geom_nx, geom_ny, geom_nz, geom_body_near`

## Geometry Toggle
In `preprocess_data.py`:
- `USE_GEOMETRY_CHANNELS = True` enables per-particle geometry channels from VTK nearest-surface queries.
- Set `False` to run state-only inputs.

## Notes
- `delta` mode: learns `x_t -> Delta x_t`.
- `ugradu` mode: learns `x_t -> [u_t, gradU_t]` and is intended for integration back into FLOWUnsteady time stepping.
- Normalization is per-channel using train split rows only.
- See `src/README.md` for full no-package-edit FLOWUnsteady integration flow.
- Package imports are split cleanly:
  - runtime runner imports `FLOWUnsteady`
  - case-specific sim file imports `FLOWVPM`/`FLOWVLM`/vehicle modules as needed.



Next Set of Tasks to update:

1. Remove geometric encoding for task1 - refer snippet [6]
2. Discuss with Hari whether normalization can be done for entire dataset or just train?
3. Fix the split between train and validate in the preprocess_data.py - it is still according to the G Sheets
4. Normalization between 



Concern:

1. The temporal resolution in task1 is that you train with data at every time step (200) without a skip and prediction happens for u and gradu at 200 time steps so that it can be inputed to the time integration at every time step without cultivating for temporal super resolution (can be tried)... temporal sr should be tried and inputed to the intergator at every time step even if particle states at every step are not given but prediction of u and grad u is done at every step, but isn't those states also required at every time step for forward integration which does not involve P2P interaction and hence does not use sr.

AFTER VERIFICATION - TEMPORAL SR IS NOT TRIVIAL HERE
(temporal sr or sr in general is done if one cannot generate that much of input-output pairs (every step), but if it requires states of particles itself which are input pair, for the time integration, then there is no point) - still think about it.

-Because, even if you avoid expensive FMM evaluations at intermediate timesteps,
you STILL need particle states at those timesteps, and those states must still be evolved. So, the expensive temporal evolution loop still exists.


2. Temporal resolution for task2 implies you are not utilizing all the processing steps for (50/200) for FMM - instead train on only certain skipped steps temporally even for training, and ask to predict at every step in between - which is temporal sr, and not training on every time step and then asking for every time step output by giving only certain skipped time step during test data.

sparse train, dense test - true temporal SR
dense train, sparse test - robustness/interpolation

Current Task 1:
replaces expensive interaction operator.

Temporal SR Task:
replaces temporal evolution itself.