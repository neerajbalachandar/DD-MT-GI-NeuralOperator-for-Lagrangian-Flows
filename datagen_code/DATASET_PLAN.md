# 39-case generation and retention plan

`./run.sh` now targets 39 cases total: 19 vortex-ring and 20 roundjet cases.
Vortex rings have 8 train, 4 validation, and 7 test cases; roundjets have 9, 4,
and 7. The opposite-sign colliding-ring case was removed after it stalled during
collision. Use `./run.sh all` to generate both families or pass `vortexrings` /
`roundjet` to run one. Output defaults to
`/media/dysco/New Volume/Neeraj/NeuralOp_FLOWVPM_Data`.

## Vortex-ring folders already present

The data drive had 27 leapfrogging folders: 19 train, 6 validate, 2 test. Their
common baseline is radius 1 and leading circulation 10. In names such as
`leapfrog_N160_d0p750_g0p500`, `N160` is nominal total particles, `d0p750` is
center spacing/radius 0.75, and `g0p500` is trailing/leading circulation ratio
0.5 (trailing circulation 5). `p` denotes a decimal point. `gm1p000` denotes a
negative ratio. See each `case.toml` for exact solver settings, save interval,
and grid resolution.

The 408-item cases each had 101 snapshots from steps 0 to 1000, saved every 10
steps. The 1007-item cases had 251 snapshots from steps 0 to 1000, saved every
4 steps. So the 408 count means a coarser save cadence, not a shorter run.
The four stride-10 folders were:

- `train/vortexrings/leapfrog_N320_d1p250_g0p500`
- `validate/vortexrings/leapfrog_N640_d0p750_g1p000`
- `validate/vortexrings/leapfrog_N640_d1p000_g1p000`
- `validate/vortexrings/leapfrog_N640_d1p250_g1p000`

The first, third, and fourth are removed by the compact design. The N640,
d/R=0.75 validation case is retained for a resolution comparison; its old
folder was removed so `run.sh` can regenerate it at stride 4. Other retained
stride-4 runs are shortened to their first 720 integration steps: snapshots
0,4,...,720 give exactly 180 adjacent transitions. The standard vortex step
size stays the same as before; the retained trajectory ends earlier rather
than using a larger timestep. The opposite-sign collision case has since been
excluded after its particle coordinates became numerically unstable. The active
plan now contains 39 cases.

## Exact existing folders to remove

These 14 redundant/intermediate leapfrogging folders are removed from the raw
data root. Their parameter values lie between retained samples, while the
retained set preserves spacing and circulation extremes and includes multiple
particle resolutions.

**Train (12):**

- `leapfrog_N160_d0p750_g1p000`
- `leapfrog_N160_d1p000_g0p500`
- `leapfrog_N160_d1p000_g1p000`
- `leapfrog_N160_d1p250_g0p500`
- `leapfrog_N160_d1p250_g0p750`
- `leapfrog_N160_d1p250_g1p000`
- `leapfrog_N160_d1p500_g0p500`
- `leapfrog_N160_d1p500_g0p750`
- `leapfrog_N160_d2p000_g0p750`
- `leapfrog_N320_d0p750_g0p500`
- `leapfrog_N320_d1p000_g0p750`
- `leapfrog_N320_d1p250_g0p500`

**Validate (2):**

- `leapfrog_N640_d1p000_g1p000`
- `leapfrog_N640_d1p250_g1p000`

The 13 existing cases retained in the active plan are (12 folders currently
remain on disk; the N640 validation folder is scheduled for regeneration):

- **Train (7):** `leapfrog_N160_d0p750_g0p500`,
  `leapfrog_N160_d1p000_g0p750`, `leapfrog_N160_d2p000_g1p000`,
  `leapfrog_N320_d0p750_g0p750`, `leapfrog_N320_d1p000_g0p500`,
  `leapfrog_N320_d1p000_g1p000`, `leapfrog_N320_d1p250_g0p750`.
- **Validate (4):** `leapfrog_N160_d0p750_g0p750`,
  `leapfrog_N160_d2p000_g0p500`, `leapfrog_N320_d0p750_g1p000`,
  `leapfrog_N640_d0p750_g1p000`.
- **Test (2):** `leapfrog_N160_d1p500_g1p000`,
  `leapfrog_N320_d1p250_g1p000`.

## Nineteen vortex-ring cases

The 13 retained cases plus these 6 additions make 19:

- **Train addition (1):** isolated ring (`N=320`, circulation 20,
  core/radius 0.08).
- **Test additions (5):** out-of-range leapfrogging (`N=320`, spacing/radius
  2.5, ratio 0.25); perturbed ring (`N=320`, amplitude/radius 0.1, mode 3,
  first 180 integration steps saved every step); matched-parameter spatial
  superresolution target on a 64-cubed grid; toroidal-vorticity ring
  (`R=1.5`, circulation 20); and a free pair (`N=320`, separation/radius 5,
  ratio -1).

The two existing tests remain in-range comparisons. The new tests cover an
outside-range operating point, an unseen initial condition, temporal and
spatial superresolution, and two unseen flow families. Vortex rings have no
solid or inlet boundary, so a boundary-condition holdout is not applicable.

## Twenty roundjet cases

The roundjet plan uses the same 9/4/7 split. Its regular cases vary diameter
`D` (1 or 2), exit speed `U` (0.75, 1, 1.25), spacing/momentum-thickness
(`dx/theta`, about 0.167 or 0.125), and overlap (2, 2.4, 3). The seven tests
include two in-range points, one Reynolds-number holdout at 15000, one faster
unseen operating point (`U=1.5`), an unseen inlet momentum-thickness condition
(`theta/D=0.04` versus the training baseline 0.025), and temporal and spatial
superresolution cases. The spatial target uses 64-cubed fields; the temporal
case saves every integration step. Standard roundjets run 720 steps (7.2
diameters at 100 steps per diameter), saved every 4 steps, for 180 transitions.

## Failure stability

The previous step-200 exception came from FLOWVPM's FMM regularization-error
root finder failing to bracket its auto-tuned root. The generator disables
that tuning and uses `rho/sigma=1.0`; the FMM solver remains active. Julia uses
four threads by default, configurable with `JULIA_NUM_THREADS`.

The opposite-sign pair `signed_N320_d2p000_gm1p000` is excluded from all future
runs. Its ring centers approached from a separation of 2 radii to about 0.16 by
step 200; by step 300, particle coordinates had blown up to magnitudes above
`1e10`, after which the FMM work stopped making progress. The partial snapshots
are preserved under `excluded_unstable/` outside the train/validation/test
splits, so preprocessing will not treat them as training data.

Completed cases at the requested cadence and duration are skipped. A retained
case at the wrong cadence, including the N640 stride-10 validation case, is
restarted from its initial condition. The simulation driver cannot resume a
trajectory mid-run.
