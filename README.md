## A Geometry-Informed, Multi-Task Neural Operator for Nonlocal Kernel Interactions in Lagrangian Flows

The architecture DD-MT-GINO is a multi-task neural operator that utilizes a shared latent space to predict two different outputs having a similar underlying PDE, using a dual decoder. This architecture is based on the GINO (Geometry-Informed Neural Operator) by Z. Li et al.

Arxiv publication:

## Current pipeline: co-moving VortexGINO (vortex rings)

One shared operator maps vortex particles `{x_p, Gamma_p, sigma_p}` to the velocity `u(q)` and
velocity gradient `J(q) = du/dq` at arbitrary points `q`. The same operator therefore

1. **reconstructs the Eulerian velocity field** when `q` are grid nodes, and
2. **drives the Lagrangian dynamics** when `q` are the particles themselves:
   `dx/dt = u(x)`, `dGamma/dt = |Gamma| r_theta` (a learned circulation rate supervised with the exact
   FLOWVPM transposed stretching `J^T Gamma / |Gamma|`), integrated with coupled RK4 substeps,
   FLOWVPM corrected-Pedrizzetti relaxation and a small learned residual.

Results, defects found in the old pipeline and all changes: `docs/IMPROVEMENT_REPORT.md`.
Final model: `runs/vortex_gino/final_model.pt`; its evaluation: `runs/vortex_gino/evaluation_final/`.

Architecture (GNO encoder -> FNO -> GNO decoder) in `gino/model/vortex_operator.py`:
- co-moving latent box (24x24x28, spacing ~0.24) centred on the circulation centroid;
- Gamma-linear, summed particle-to-grid encoder (superposition, invariant to particle resolution);
- neuralop FNO with domain padding;
- decoder = latent sample + equivariant learned near-field Biot-Savart kernel with an analytic gradient.

End to end (data -> tests -> training -> evaluation + plots), ~4-8 h on a 16 GB GPU:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash scripts/run_full_pipeline.sh            # uses configs/vortex_gino_full.yaml, writes runs/vortex_gino_full/
RESUME=1 bash scripts/run_full_pipeline.sh   # continue after an interruption
```

Individual steps:

```bash
# 1. lossless per-case archives from the raw FLOWVPM output (all particles, exact particle u/J,
#    exact Biot-Savart grid velocity, FLOWVPM vorticity)            (~8 min, GPU)
python scripts/build_vortex_dataset.py --raw-root datagen_code/training_data --out processed_data/vortex_rings_v2
# 2. train (operator phase, then joint operator + multi-step RK4 dynamics)
python scripts/train_vortex.py --config configs/vortex_gino.yaml
# 3. evaluate: teacher-forced field reconstruction + 180-step rollouts + plots + summary.md
python scripts/evaluate_vortex.py --checkpoint runs/vortex_gino/final_model.pt
# tests
python -m pytest tests/test_vortex_operator.py tests/test_vortex_data.py
```

Checkpoints store EMA weights in `model_state_dict` (used for validation and evaluation) and the
raw optimizer iterate in `raw_model_state_dict` (used by `--resume`).

The legacy pipeline (`scripts/preprocess.py`, `scripts/train.py`, `scripts/evaluate.py`,
`gino/model/gino.py`) is kept for reference; see `docs/REPOSITORY_GUIDE.md` for the list of defects
that were found in it and why the new pipeline replaced it.
