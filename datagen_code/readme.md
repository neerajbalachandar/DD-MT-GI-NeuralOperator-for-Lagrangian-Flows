# VPM dataset generation

Run `./run.sh all` to generate the planned 40 cases: 20 vortex-ring and 20
roundjet simulations, each split into 9 train, 4 validation, and 7 test cases.
Use `./run.sh vortexrings` or `./run.sh roundjet` to generate one family. The
output root defaults to `/media/dysco/New Volume/Neeraj/NeuralOp_FLOWVPM_Data`;
set `OUTPUT_ROOT` to change it.

Standard cases save every 4 integration steps through 720 steps, producing 181
snapshots (180 adjacent input-target pairs). The temporal-superresolution cases
save each of their first 180 integration steps. Preprocessing defaults to 180
pairs per case and uses an uninterrupted beginning-of-trajectory prefix.

The case selection, exact parameter values, retained folders, removed folders,
and evaluation categories are listed in [DATASET_PLAN.md](DATASET_PLAN.md).
The external root receives an updated `case_plan.toml` when `run.sh` starts.
Completed cases with the planned cadence and duration are skipped. Incomplete
or wrong-cadence cases are restarted from their initial condition.
