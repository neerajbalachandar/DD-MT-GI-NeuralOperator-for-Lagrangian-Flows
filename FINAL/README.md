# Shared Latent VPM GINO

This directory is the modular research implementation for the VPM GINO workflow. The canonical scientific definitions remain those in the repository's `process_data.py`, `GINO_sharedlatent.ipynb`, and `GINO_analysis.py`. The refactor preserves the seven-channel physical state `[x,y,z,Gamma_x,Gamma_y,Gamma_z,sigma]`, the source input feature order, the Euler VPM base transition, and the shared GNO/FNO latent with particle and field decoders.

## Commands

```bash
python scripts/preprocess.py
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --set training.use_rollout_loss=true --set training.rollout_weight=1.0 --set training.rollout_horizon_max=16 --set training.rollout_horizon_schedule='[1, 2, 4, 8, 16]'
python scripts/evaluate.py --config configs/default.yaml --set evaluation.checkpoint=runs/baseline/best_model.pt
```

Override any config value using `--set section.key=value`. Training requires a preprocessed NPZ with whole-case split metadata and a nonempty validation case split. It records resolved config, environment, history, split IDs, normalization, and mechanism flags beside the checkpoint. Validation selects checkpoints; test data is not used for selection.

After the physical-spacing gradient and VTK-context corrections, rerun preprocessing from the mounted Task-1/Task-2 sources before training. Older NPZ files may lack frame-specific VTK paths and cannot support geometry-correct autoregressive reconstruction.

## Dependency flow

`scripts` → `gino.data` + `gino.model` + `gino.dynamics` + `gino.training` / `gino.evaluation` → `visualization`.

`state_transition.predict_next_state` is the single physical-plus-residual step and exposes predicted/base states, residual, and normalized/physical fields. Inference uses `inference_rollout`; BPTT uses `training_rollout`; detached generated-input training uses `pushforward_rollout`. All share `gino.data.reconstruction.rebuild_next_batch`, which updates state, flow, phase, and VTK-derived geometry inputs.

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

The baseline keeps the original state-dict layer names and shapes (`lift`, `global_condition_mlp`, `encoder`, `fno`, decoder and attention projection names, and uncertainty scalars). Evaluation checks missing and unexpected checkpoint keys before running.

## Mechanisms

Baseline: one-step state residual and field losses with the existing homoscedastic task weighting. Mechanisms are independent config switches. During generated multi-step training, scheduled sampling makes one decision for each new input and replaces only velocity/gradient features in `x`; it does not turn on rollout training by itself. GNS noise is pre-generated per rollout transition and added in normalized input-feature space. Pushforward detaches generated inputs between supervised steps, while rollout loss retains gradients through generated states.

The BPTT path is not fully differentiable through geometry: predicted positions are detached for the VTK/NumPy nearest-surface feature lookup. State, field, and normalized-coordinate paths retain gradients, but geometry-derived inputs do not. Preprocessing matches particles by stable IDs when an ID dataset is available. When source HDF5 has no IDs, tagged source-row indices are used and the archive context records that fallback; it cannot detect a physical particle reorder. Evaluation requires normalization statistics from the checkpoint and does not fall back to global dataset statistics.

Evaluation reports one-step physical state errors, autoregressive metrics by configured horizon, and direct native Task-2 HDF5 field errors when `data.native_task2_root` is set. Autoregressive evaluation never teacher-forces future states. Preprocessing stores source VTK paths so geometry-dependent channels can be refreshed at every rollout step.
