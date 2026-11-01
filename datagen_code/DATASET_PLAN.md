# VPM training dataset plan

`run.sh` writes to `/media/dysco/New Volume/Neeraj/NeuralOp_FLOWVPM_Data` by default. The generated root contains `train/`, `validate/`, and `test/`, each split by `vortexrings/` and `roundjet/`. `case_plan.toml` records the selected cases, split, evaluation categories, grid size, and save cadence.

## Case budget

The systematic plan contains 120 cases: 98 vortex-ring cases and 22 roundjet cases. The case split is 84 train, 18 validation, and 18 test (70/15/15). Existing complete folders are retained in their current split. The selected families are:

| Family | Cases |
| --- | ---: |
| Leapfrogging rings | 27 |
| Isolated rings | 16 |
| Signed ring pairs | 14 |
| Toroidal rings | 16 |
| Free ring pairs | 16 |
| Perturbed rings | 9 |
| Roundjet, standard operating range | 20 |
| Roundjet, held-out Reynolds number | 2 |

The parameter candidates retain three ring resolutions (160, 320, 640 where supported), multiple circulation/spacing/core-size settings, and roundjet diameters 1 and 2 with exit speeds 0.75, 1.0, and 1.25. The sweep samples these parameter grids evenly and stratifies validation cases across families.

Standard cases save every 10 simulation steps and preprocessing keeps at most 100 transitions per case. It selects frames by farthest-point sampling over particle position, circulation, and velocity summaries, preserving both endpoints and reducing repeated flow states. The temporal superresolution case saves every step and keeps its dense sequence. All standard field grids are 32 cubed; the spatial superresolution case is generated at 64 cubed and compared on its native grid during evaluation.

## Test protocol

* **In regime:** the two existing held-out leapfrog cases `leapfrog_N160_d1p500_g1p000` and `leapfrog_N320_d1p250_g1p000` remain test cases. On a clean output root, two cases inside the selected training ranges fill this category instead.
* **Outside the selected toroidal radius range:** five toroidal cases use radius 1.25; train/validation toroidal cases use radii 0.75 and 1.0.
* **Unseen operation/Reynolds number:** two roundjets use Reynolds number 15,000; the remaining roundjet cases use 10,000. Their diameter, speed, spacing, and overlap are recorded in `case.toml`.
* **Unseen initial conditions:** nine perturbed-ring configurations are held out from training. This unbounded vortex setup has no solid boundary condition to hold out.
* **Temporal superresolution:** one of the perturbed test cases saves every step, while standard cases save every 10 steps. Both particle inputs and simulation field outputs are retained.
* **Spatial superresolution:** one held-out Reynolds-number roundjet uses a 64-cubed field grid. The model is trained on 32-cubed field samples and evaluation scores predictions at all native 64-cubed nodes.

Every test transition retains the particle state at time `t`, the target state at `t+1`, and its aligned field snapshot. Test cases are never split by frame. Evaluation writes one-step records with their scenario tags and native-field metrics grouped by category.

## Power-failure recovery

The output present when this plan was added contains 18 train, 3 validate, and 2 test leapfrog folders. The interrupted next leapfrog configuration is `leapfrog_N320_d1p250_g0p500`. The sweep includes it explicitly. Complete cases are skipped; a directory missing any expected particle or field HDF5/XMF snapshot is removed and regenerated. Preprocessing independently skips cases with missing snapshots.
