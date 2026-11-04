# VPM dataset generation

Run `./run.sh all` to generate the planned 39 cases: 19 vortex-ring and 20
roundjet simulations. Vortex rings have 8 train, 4 validation, and 7 test cases;
roundjets have 9, 4, and 7. The opposite-sign colliding-ring case was removed
because its integration stalls during collision.
Use `./run.sh vortexrings` or `./run.sh roundjet` to generate one family. The
output root defaults to `/media/dysco/New Volume/Neeraj/NeuralOp_FLOWVPM_Data`;
set `OUTPUT_ROOT` to change it.

Vortex-ring standard cases save every 4 steps through 720 steps. Roundjet
standard cases use a coarser 50-step-per-diameter integrator and save every 2
steps through 360 steps, keeping 181 snapshots (180 adjacent input-target
pairs) at the same physical-time cadence. The roundjet temporal-superresolution
case saves each integration step over the same 360-step trajectory. Preprocessing
defaults to 180 pairs per case and uses an uninterrupted beginning-of-trajectory
prefix.

The case selection, exact parameter values, retained folders, removed folders,
and evaluation categories are listed in [DATASET_PLAN.md](DATASET_PLAN.md).
The external root receives an updated `case_plan.toml` when `run.sh` starts.
Completed cases with the planned cadence and duration are skipped. Interrupted
vortex-ring cases resume from the latest valid particle checkpoint.

Roundjet generation uses analytically integrated annulus circulation with the
same 0.5 calibration as the RBF target by default. The iterative RBF
conjugate-gradient fit took about eight minutes per operator evaluation for a
112,000-particle case and allows up to 200 CG iterations, so it is opt-in
with `ROUNDJET_RBF=1`. Fit progress and the configured timeout remain enabled in
that mode.

Vortex-ring particle counts are invariant after initialization; the driver
checks this after every time-step callback. Roundjets now start as finite
particle pulses (`ROUNDJET_PULSE_D`, default `0.10D`) with no runtime inlet.
Their particle count is fixed for the whole trajectory, and each case records
the initial and final count in `case.toml`. Their output folders include the
pulse length (for example, `_pulse0p100D`) so they do not overwrite older
continuous-inlet datasets. This represents a finite-duration jet puff, not a
steady jet. Increase `ROUNDJET_PULSE_D` for a longer puff, with a corresponding
increase in memory and compute cost.

For roundjets, the core-spreading reset threshold is set just above the maximum
viscous core growth predicted over the configured trajectory. This preserves
the intended viscous core growth while avoiding the expensive global RBF
strength reset that previously caused long apparent stalls. The pulse length
and `core_spreading_beta` are saved in each case's metadata.

Roundjet sweep particle spacing is now coarser (`dx/theta=1/3` or `1/4`, rather
than `1/6` or `1/8`). This lowers the annulus particle count substantially while
keeping 180 standard saved transitions. The temporal test remains at every
integration step, and the spatial test still evaluates fields on a 64-cubed
grid.
