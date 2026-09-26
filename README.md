## A Geometry-Informed, Multi-Task Neural Operator for Nonlocal Kernel Interactions in Lagrangian Flows

The architecture DD-MT-GINO is a multi-task neural operator that utilizes a shared latent space to predict two different outputs having a similar underlying PDE, using a dual decoder. This architecture is based on the GINO (Geometry-Informed Neural Operator) by Z. Li et al.

Arxiv publication:


The current issues:

Preprocess the raw Task-1 particles and Task-2 field data with:

```bash
python3 scripts/preprocess.py
```

By default, the CLI reads `/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task1` and `/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task2`, and writes the canonical archive to `processed_data/particle_evolution_dataset.npz`. Override these locations with `--source-root`, `--field-root`, or `--output-dir`. The pipeline is implemented in `gino/data/preprocessing_pipeline.py`; no root-level `process_data.py` module is needed.
