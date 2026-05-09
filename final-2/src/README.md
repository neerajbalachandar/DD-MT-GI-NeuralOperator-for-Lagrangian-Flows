# final-2/src (FLOWUnsteady Integration, No Package Edits)

This folder provides a **standalone integration layer** so you do not need to edit installed FLOWUnsteady/FLOWVPM source files.

## What Is Here
- `task1_ugradu_settings.jl`
- `task1_particle_io.jl`
- `task1_python_predictor.jl`
- `task1_python_predictor.py`
- `task1_flowunsteady_runtime.jl`
- `run_task1_flowunsteady.jl`
- `task1_simfile_template.jl`
- legacy templates:
  - `flowunsteady_export_ugradu.jl`
  - `flowunsteady_ml_bridge.jl`

## Input Channels Used For Task-1-u/gradU (Current)
By default (metadata-free):
- `x`
- `y`
- `z`
- `Gamma_x`
- `Gamma_y`
- `Gamma_z`
- `sigma`
- geometry channels (if enabled in preprocessing and model):
  - `geom_dist`
  - `geom_nx`
  - `geom_ny`
  - `geom_nz`
  - `geom_body_near`

Optional context channels (only if enabled in preprocessing/model):
- `phase`
- `angle_of_attack`
- `freestream_x`
- `freestream_y`
- `freestream_z`

## Target Channels
- `velocity_x`, `velocity_y`, `velocity_z`
- `gradUx_x`, `gradUx_y`, `gradUx_z`
- `gradUy_x`, `gradUy_y`, `gradUy_z`
- `gradUz_x`, `gradUz_y`, `gradUz_z`

## Runtime Modes
Configured in `UGradUSettings.mode`:
- `:baseline_fmm`   -> normal FLOWUnsteady FMM/solver U/gradU
- `:surrogate_ml`   -> replace U/gradU with ML model
- `:compare`        -> run surrogate and save residuals vs baseline arrays

## How To Use
1. Train model via `task1_particle_field_evolution_2.ipynb`.
2. Build dataset with `TASK1_TARGET_MODE = "ugradu"` in `preprocess_data.py`.
3. Prepare your simulation file with:
   - `build_task1_simulation() -> (sim, run_kwargs::Dict)`
   - import your case dependencies there (`using FLOWUnsteady`, `using FLOWVPM`, `using FLOWVLM`, plus your own includes)
   - use `task1_simfile_template.jl` as starting point
4. Run:
   - `julia --project final-2/src/run_task1_flowunsteady.jl --sim-file /abs/path/to/your_sim_file.jl --mode compare`

## Important
- FLOWUnsteady APIs differ by version.
- `task1_particle_io.jl` contains the only places where pfield member names are assumed (`X`, `Gamma`, `sigma`, `U`, `J`).
- If your version differs, edit only that file.

## Where Packages Are Imported
- `run_task1_flowunsteady.jl` imports `FLOWUnsteady` and calls `FLOWUnsteady.run_simulation`.
- Your simulation file (passed with `--sim-file`) should import any extra packages needed by that specific case (`FLOWVPM`, `FLOWVLM`, vehicle/maneuver modules, etc.).
- This keeps the runtime pipeline unified while allowing case-specific dependencies to remain local to your sim file.
