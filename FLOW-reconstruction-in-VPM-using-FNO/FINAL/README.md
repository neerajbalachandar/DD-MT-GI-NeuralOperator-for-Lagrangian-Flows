# Shared Latent VPM GINO

This directory is the modular research implementation for the VPM GINO workflow. The canonical scientific definitions remain those in the repository's `process_data.py`, `GINO_sharedlatent.ipynb`, and `GINO_analysis.py`. The refactor preserves the seven-channel physical state `[x,y,z,Gamma_x,Gamma_y,Gamma_z,sigma]`, the source input feature order, the Euler VPM base transition, and the shared GNO/FNO latent with particle and field decoders.

## Commands

```bash
python scripts/preprocess.py --source-root ../..
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --set evaluation.checkpoint=runs/baseline/best_model.pt
```

Override any config value using `--set section.key=value`. Training requires a preprocessed NPZ with whole-case split metadata and a nonempty validation case split. It records resolved config, environment, history, split IDs, normalization, and mechanism flags beside the checkpoint. Validation selects checkpoints; test data is not used for selection.

## Dependency flow

`scripts` → `gino.data` + `gino.model` + `gino.dynamics` + `gino.training` / `gino.evaluation` → `visualization`.

`state_transition.predict_next_state` is the single physical-plus-residual step. `dynamics.rollout` always delegates the next input to a caller supplied `rebuild_batch` function so geometry must be recalculated from each predicted position state.

## Source mapping

| Existing implementation | Modular destination |
| --- | --- |
| `process_data.py`: HDF5 reads and vector parsing | `gino/data/hdf5.py` |
| `process_data.py`: VTK geometry distance, normals, near-body channel | `gino/data/geometry.py` |
| `process_data.py`: state/feature order, residual and delta targets, train statistics | `gino/data/preprocessing.py`, `gino/data/dataset.py`, `gino/data/normalization.py` |
| `GINO_sharedlatent.ipynb`: GNO, FNO, global conditioning, positional encoding, shared representation, both decoders | `gino/model/` |
| `GINO_sharedlatent.ipynb`: uncertainty weighted losses, scheduled-sampling behavior, horizon curriculum | `gino/training/` |
| `GINO_analysis.py`: Euler base plus residual transition and rollout evaluation | `gino/dynamics/`, `gino/evaluation/` |
| analysis figures | `visualization/` |

The baseline keeps the original state-dict layer names and shapes (`lift`, `global_condition_mlp`, `encoder`, `fno`, decoder and attention projection names, and uncertainty scalars). Checkpoint compatibility still requires an actual checkpoint load test; this workspace contains no `.pt`/`.pth` file, processed dataset, or raw HDF5 data, and `torch` is absent from the active Python runtime.

## Mechanisms

Baseline: one-step state residual and field losses with the existing homoscedastic task weighting. Optional modules expose scheduled sampling, temporally correlated GNS noise, pushforward inputs, normalized rollout loss, longer horizon curriculum, attention/skip/global-conditioning switches, and task adapters. New mechanisms default off. Scheduled sampling mirrors the source notebook's partial velocity/gradient replacement and is not equivalent to closed-loop rollout.

Numerical one-step evaluation is available from the CLI. Autoregressive evaluation accepts a `rebuild_batch` callback and never teacher-forces. This callback must refresh all state-dependent features, including geometry. Native Task-2 field metrics operate on native HDF5 samples; plotting interpolation is separate.
