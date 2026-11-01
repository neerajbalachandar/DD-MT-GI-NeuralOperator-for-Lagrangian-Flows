- The generator checks for every expected particle and field HDF5/XMF snapshot. It skips complete cases and deletes and regenerates incomplete ones.
- From the folders shown, the missing next leapfrog case is leapfrog_N320_d1p250_g0p500. It is explicitly included in the plan. The 23 existing folders—18 train, 3 validate, 2 test—are retained.
Smaller, balanced dataset
The plan selects 120 cases: 98 vortex-ring cases and 22 roundjet cases. The split is 84 train, 18 validate, 18 test. Family counts and test categories are recorded in [DATASET_PLAN.md](/home/dysco/FLOWUnsteady/Flow-reconstruction-in-VPM-using-FNO/datagen_code/DATASET_PLAN.md) and in the generated case_plan.toml.
Standard cases save every 10 steps; preprocessing keeps up to 100 transitions per case, selecting frames by particle and flow summaries to reduce repeated states. The temporal superresolution test case saves every step and keeps its dense sequence.
Test scenarios
The 18 test cases cover:
- In-regime: the two existing held-out leapfrog cases, leapfrog_N160_d1p500_g1p000 and leapfrog_N320_d1p250_g1p000.
- Outside the selected toroidal radius range: five cases at radius 1.25; training and validation use radii 0.75 and 1.0.
- Unseen operating condition: two roundjets at Reynolds number 15,000; the other roundjets use 10,000.
- Unseen initial conditions: nine perturbed-ring cases, including the temporal superresolution case.
- Temporal superresolution: one perturbed test case saved every step, compared with standard cases saved every 10 steps.
- Spatial superresolution: one held-out roundjet uses a 64³ field grid; evaluation scores predictions at all its native nodes.
Test particle inputs, next-step targets, and aligned field snapshots are included in preprocessing. Incomplete cases are excluded. Training logs the held-out cases and their categories; evaluation writes category-tagged pair records and native-field summaries.
I also updated preprocessing, training and evaluation, plus the data inspection, training inspection, result analysis, and visualization notebooks. The dataset plan documents the selections and resume behavior. Start with ./run.sh all; it will skip complete cases and generate the remaining selected cases.