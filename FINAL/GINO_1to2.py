#!/usr/bin/env python
# coding: utf-8

# # Updated GINO model - 23 June 2026
# 
# Targets - $u_x, u_y, u_z$, $\nabla{u}_x, \nabla{u}_y, \nabla{u}_z$
# 
# 
# From the predicted particle velocity - use either sr, or a pointwise decoder from latent space for reconstructing flow field. Use the predicted values to propogate the states

# Unified GINO training for particle U/gradU and Eulerian reconstruction.
# 
# Modes
# -----
# GINO_TASK=particle_ugradu
#     Particle locations are both input geometry and output queries. The model
#     predicts [U, gradU] at particles for FLOWUnsteady/rVPM integration.
# 
# GINO_TASK=field_reconstruction
#     Particle states are mapped to an Eulerian velocity/vorticity field.
#     GINO_FIELD_DECODER=super_resolution uses the normal NeuralOperator
#     GINO output GNO decoder. GINO_FIELD_DECODER=pointwise uses a
#     pointwise query decoder over a learned latent grid.
# 
# Channel sweeps
# --------------
# GINO_INPUT_CHANNELS="Gamma_x,Gamma_y,Gamma_z,sigma,geom_dist,angle_of_attack,phase"
# selects exact processed-data feature names. If unset, the defaults below are
# used. Coordinates are always supplied through `input_geom`; include x/y/z as
# feature channels only if you explicitly want them duplicated.

# In[1]:
from __future__ import annotations

import socket

print(socket.gethostname())


# Include processing and visualizing data snippets from final-2/task1 notebook.
# Check if all the required parameters are saved in the .pt file and required plots should be put in a new file, including loading the trained model. 

# In[ ]:

import inspect
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from neuralop.models import GINO
    GINO_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - depends on runtime env
    GINO = None
    GINO_IMPORT_ERROR = error

try:
    from neuralop.layers.gno_block import GNOBlock
    GNOBLOCK_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - depends on runtime env
    GNOBlock = None
    GNOBLOCK_IMPORT_ERROR = error

try:
    from neuralop.utils import count_model_params
except Exception:  # pragma: no cover
    def count_model_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

print('Torch       :', torch.__version__)
print('CUDA build  :', torch.version.cuda)
print('CUDA usable :', torch.cuda.is_available())


# In[ ]:


SEED = int(os.environ.get("GINO_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_or_none(raw: str) -> Optional[int]:
    raw = str(raw).strip().lower()
    if raw in {"", "none", "null", "all", "0"}:
        return None
    return int(raw)


def _csv_env(name: str, default: Sequence[str]) -> List[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _safe_np_load(path: Path):
    """Load datasets saved by newer NumPy from older system NumPy when needed."""
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError as error:
        if "numpy._core" not in str(error):
            raise
        import numpy.core as numpy_core

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        return np.load(path, allow_pickle=True)


def _repo_paths() -> Tuple[Path, Path]:
    cwd = Path.cwd().resolve()
    if (cwd / "FINAL").is_dir():
        return cwd, cwd / "FINAL"
    if cwd.name == "FINAL":
        return cwd.parent, cwd
    if cwd.parent.name == "FINAL":
        return cwd.parent.parent, cwd.parent
    if (cwd / "final-2").is_dir():
        return cwd, cwd / "final-2"
    if cwd.name == "final-2":
        return cwd.parent, cwd
    if cwd.parent.name == "final-2":
        return cwd.parent.parent, cwd.parent
    return cwd, cwd / "final-2"


REPO_ROOT, FINAL_DIR = _repo_paths()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK = os.environ.get("GINO_TASK", "particle_ugradu").strip().lower()
FIELD_DECODER = os.environ.get("GINO_FIELD_DECODER", "super_resolution").strip().lower()
if TASK not in {"particle_ugradu", "field_reconstruction"}:
    raise ValueError("GINO_TASK must be particle_ugradu or field_reconstruction")
if FIELD_DECODER not in {"super_resolution", "pointwise"}:
    raise ValueError("GINO_FIELD_DECODER must be super_resolution or pointwise")

print(REPO_ROOT, FINAL_DIR, DEVICE)


# In[ ]:


DEFAULT_PARTICLE_CHANNELS = [
    "Gamma_x",
    "Gamma_y",
    "Gamma_z",
    "sigma",
    "geom_dist",
    "geom_nx",
    "geom_ny",
    "geom_nz",
    "angle_of_attack",
    "phase",
]
DEFAULT_FIELD_CHANNELS = DEFAULT_PARTICLE_CHANNELS


# In[ ]:


# CFG = {
#     "seed": SEED,
#     "task": TASK,
#     "field_decoder": FIELD_DECODER,
#     "file_tag": os.environ.get(
#         "GINO_FILE_TAG",
#         "task1_particle_ugradu_gino" if TASK == "particle_ugradu" else f"task2_gino_{FIELD_DECODER}",
#     ),
#     "epochs": int(os.environ.get("GINO_EPOCHS", "60")),
#     "lr": float(os.environ.get("GINO_LR", "3e-4")),
#     "weight_decay": float(os.environ.get("GINO_WEIGHT_DECAY", "1e-4")),
#     "eval_every": int(os.environ.get("GINO_EVAL_EVERY", "5")),
#     "gradient_accumulation_steps": int(os.environ.get("GINO_ACCUM_STEPS", "4")),
#     "grad_clip_norm": float(os.environ.get("GINO_GRAD_CLIP", "1.0")),
#     "maximum_input_particles": _int_or_none(os.environ.get("GINO_MAX_INPUT_PARTICLES", "2000")),
#     "maximum_train_output_points": _int_or_none(os.environ.get("GINO_MAX_TRAIN_OUTPUT_POINTS", "2048")),
#     "maximum_eval_output_points": _int_or_none(os.environ.get("GINO_MAX_EVAL_OUTPUT_POINTS", "32768")),
#     "batch_size": 1,
#     "num_workers": int(os.environ.get("GINO_NUM_WORKERS", "0")),
#     "use_amp": _bool_env("GINO_USE_AMP", True),
#     "latent_res": int(os.environ.get("GINO_LATENT_RES", "16")),
#     "in_gno_radius": float(os.environ.get("GINO_IN_RADIUS", "0.35")),
#     "out_gno_radius": float(os.environ.get("GINO_OUT_RADIUS", "0.40")),
#     "in_gno_transform_type": os.environ.get("GINO_IN_TRANSFORM", "nonlinear_kernelonly"),
#     "out_gno_transform_type": os.environ.get("GINO_OUT_TRANSFORM", "linear"),
#     "gno_embed_channels": int(os.environ.get("GINO_EMBED_CHANNELS", "32")),
#     "fno_n_modes": tuple(int(x) for x in os.environ.get("GINO_FNO_MODES", "4,4,4").split(",")),
#     "fno_hidden_channels": int(os.environ.get("GINO_FNO_HIDDEN", "32")),
#     "fno_n_layers": int(os.environ.get("GINO_FNO_LAYERS", "4")),
#     "projection_channel_ratio": int(os.environ.get("GINO_PROJ_RATIO", "2")),
#     "gno_use_open3d": _bool_env("GINO_USE_OPEN3D", True),
#     "gno_use_torch_scatter": _bool_env("GINO_USE_TORCH_SCATTER", True),
#     "pointwise_hidden": int(os.environ.get("GINO_POINTWISE_HIDDEN", "96")),
#     "pointwise_layers": int(os.environ.get("GINO_POINTWISE_LAYERS", "3")),
# }


CFG = {
    # ===== Basic =====
    "seed": SEED,
    "task": "particle_ugradu",                      # force particle task
    "field_decoder": "pointwise",                   # lightweight decoder (no output GNO)
    "file_tag": "task1_particle_ugradu_gino_pointwise",

    # ===== Training =====
    "epochs": 30,                                   # quick test; increase later
    "lr": 1e-3,                                     # slightly higher for faster start
    "weight_decay": 1e-4,
    "eval_every": 5,
    "gradient_accumulation_steps": 4,
    "grad_clip_norm": 1.0,
    "batch_size": 1,
    "num_workers": 0,

    # ===== Sampling (memory & speed) =====
    "maximum_input_particles": 1000,                # was 2000
    "maximum_train_output_points": 512,             # was 2048
    "maximum_eval_output_points": 4096,             # enough for validation

    # ===== Mixed precision (disable to avoid complex‑dtype issues) =====
    "use_amp": False,                               # critical

    # ===== GINO architecture =====
    "latent_res": 8,                                # was 16 → 8× less memory
    "in_gno_radius": 0.35,
    "out_gno_radius": 0.40,                         # not used for pointwise, but kept
    "in_gno_transform_type": "nonlinear_kernelonly",
    "out_gno_transform_type": "linear",
    "gno_embed_channels": 32,
    "fno_n_modes": (4, 4, 4),
    "fno_hidden_channels": 32,
    "fno_n_layers": 4,
    "projection_channel_ratio": 2,

    # ===== Dependencies (disable to avoid install issues) =====
    "gno_use_open3d": False,
    "gno_use_torch_scatter": False,

    # ===== Pointwise decoder (only used when field_decoder='pointwise') =====
    "pointwise_hidden": 96,
    "pointwise_layers": 3,
}


# In[ ]:


def _find_task2_dataset() -> Path:
    env = os.environ.get("BACKUP_TASK_DATASET", "").strip()
    candidates = [Path(env)] if env else []
    candidates += [
        FINAL_DIR / "processed_data" / "Backup Data"/ "task2_fdom_dataset.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find task2_fdom_dataset.npz")


def _find_particle_dataset() -> Path:
    env = os.environ.get("FINAL2_TASK1_UGRADU_DATASET", "").strip()
    candidates = [Path(env)] if env else []
    candidates += [
        FINAL_DIR / "processed_data" / "particle_ugradu_dataset.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find particle_ugradu_dataset.npz."
    )


DATASET_PATH = _find_particle_dataset() if TASK == "particle_ugradu" else _find_task2_dataset()
RESULTS_DIR = FINAL_DIR / "result" / ("task1" if TASK == "particle_ugradu" else "task2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = RESULTS_DIR / f"{CFG['file_tag']}_best_model.pt"
LAST_CKPT_PATH = RESULTS_DIR / f"{CFG['file_tag']}_last_model.pt"
HISTORY_PATH = RESULTS_DIR / f"{CFG['file_tag']}_history.json"


# In[ ]:


def portable_path(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(path)


def sample_indices(n: int, cap: Optional[int], seed: int) -> np.ndarray:
    if cap is None or cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(cap), replace=False)).astype(np.int64)


def as_context(obj) -> Dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        item = obj.item()
        if isinstance(item, dict):
            return item
    return dict(obj)


def split_ids(ds, key: str) -> np.ndarray:
    if key in ds.files:
        return np.asarray(ds[key], dtype=np.int64)
    return np.zeros((0,), dtype=np.int64)


# In[ ]:


class BaseGraphDataset(Dataset):
    feature_names_all: List[str]
    target_names: List[str]

    def __init__(self, frame_ids: np.ndarray, split_name: str, max_input_particles: Optional[int], max_output_points: Optional[int]):
        self.frame_ids = np.asarray(frame_ids, dtype=np.int64)
        self.split_name = split_name
        self.max_input_particles = max_input_particles
        self.max_output_points = max_output_points

    def __len__(self) -> int:
        return int(len(self.frame_ids))


class ParticleUGradUDataset(BaseGraphDataset):
    def __init__(self, ds, active_indices: Sequence[int], frame_ids: np.ndarray, split_name: str, max_input_particles: Optional[int], max_output_points: Optional[int]):
        super().__init__(frame_ids, split_name, max_input_particles, max_output_points)
        self.inputs = ds["inputs_by_frame"]
        self.targets = ds["targets_by_frame"]
        self.contexts = list(ds["frame_contexts"])
        self.active_indices = np.asarray(active_indices, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict:
        frame_id = int(self.frame_ids[int(idx)])
        features_all = np.asarray(self.inputs[frame_id], dtype=np.float32)
        targets_all = np.asarray(self.targets[frame_id], dtype=np.float32)
        n = min(features_all.shape[0], targets_all.shape[0])
        in_idx = sample_indices(n, self.max_input_particles, CFG["seed"] + frame_id)
        out_idx = sample_indices(len(in_idx), self.max_output_points, CFG["seed"] + 100000 + frame_id)
        query_idx = in_idx[out_idx]

        input_features = features_all[in_idx]
        query_features = features_all[query_idx]

        input_geom = normalize_xyz(input_features[:, coord_feature_indices])
        output_queries = normalize_xyz(query_features[:, coord_feature_indices])
        x = (input_features[:, self.active_indices] - input_mean[None, :]) / input_std[None, :]
        y = (targets_all[query_idx] - target_mean[None, :]) / target_std[None, :]
        return {
            "input_geom": torch.from_numpy(input_geom),
            "x": torch.from_numpy(np.clip(np.nan_to_num(x), -8.0, 8.0).astype(np.float32)),
            "output_queries": torch.from_numpy(output_queries),
            "y": torch.from_numpy(np.nan_to_num(y).astype(np.float32)),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
        }


class FieldReconstructionDataset(BaseGraphDataset):
    def __init__(self, sample_paths: Sequence[Path], active_indices: Sequence[int], frame_ids: np.ndarray, split_name: str, max_input_particles: Optional[int], max_output_points: Optional[int]):
        super().__init__(frame_ids, split_name, max_input_particles, max_output_points)
        self.sample_paths = list(sample_paths)
        self.active_indices = np.asarray(active_indices, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict:
        frame_id = int(self.frame_ids[int(idx)])
        with _safe_np_load(self.sample_paths[frame_id]) as d:
            input_geom_raw = np.asarray(d["input_geom"], dtype=np.float32)
            features_all = np.asarray(d["input_features"], dtype=np.float32)
            queries_raw = np.asarray(d["output_queries"], dtype=np.float32)
            targets_raw = np.nan_to_num(np.asarray(d["targets"], dtype=np.float32))
        in_idx = sample_indices(input_geom_raw.shape[0], self.max_input_particles, CFG["seed"] + frame_id)
        out_idx = sample_indices(queries_raw.shape[0], self.max_output_points, CFG["seed"] + 100000 + frame_id)
        x = (features_all[in_idx][:, self.active_indices] - input_mean[None, :]) / input_std[None, :]
        y = (targets_raw[out_idx] - target_mean[None, :]) / target_std[None, :]
        return {
            "input_geom": torch.from_numpy(normalize_xyz(input_geom_raw[in_idx])),
            "x": torch.from_numpy(np.clip(np.nan_to_num(x), -8.0, 8.0).astype(np.float32)),
            "output_queries": torch.from_numpy(normalize_xyz(queries_raw[out_idx])),
            "y": torch.from_numpy(np.nan_to_num(y).astype(np.float32)),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
        }


# In[ ]:


def resolve_sample_path(path_like, base: Path) -> Path:
    raw = Path(str(path_like))
    candidates = [raw] if raw.is_absolute() else []
    candidates += [base / raw, base / "task2_gino_frames" / raw.name, FINAL_DIR / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing sample path {raw}; checked {candidates}")


dataset_file = _safe_np_load(DATASET_PATH)


# In[ ]:


if TASK == "particle_ugradu":
    feature_names_all = [str(x) for x in dataset_file["feature_names"].tolist()]
    target_names = [str(x) for x in dataset_file["target_names"].tolist()]
    requested_channels = _csv_env("FINAL2_GINO_INPUT_CHANNELS", DEFAULT_PARTICLE_CHANNELS)
    train_frame_ids = split_ids(dataset_file, "train_frame_ids")
    val_id_frame_ids = split_ids(dataset_file, "val_id_frame_ids")
    val_angle_frame_ids = split_ids(dataset_file, "validation_angle_frame_ids")
    if len(val_angle_frame_ids) == 0:
        val_angle_frame_ids = split_ids(dataset_file, "val_frame_ids")
    test_normal_frame_ids = split_ids(dataset_file, "test_normal_frame_ids")
    if len(test_normal_frame_ids) == 0:
        test_normal_frame_ids = split_ids(dataset_file, "test_frame_ids")
    test_spatial_sr_frame_ids = split_ids(dataset_file, "test_super_resolution_frame_ids")
    test_temporal_sr_frame_ids = np.zeros((0,), dtype=np.int64)
    test_unseen_angle_frame_ids = split_ids(dataset_file, "test_unseen_angle_frame_ids")
    coord_feature_indices = np.asarray([feature_names_all.index(k) for k in ("x", "y", "z")], dtype=np.int64)
    if "coord_min" in dataset_file.files and "coord_span" in dataset_file.files:
        coord_min = np.asarray(dataset_file["coord_min"], dtype=np.float32).reshape(3)
        coord_span = np.asarray(dataset_file["coord_span"], dtype=np.float32).reshape(3)
    else:
        xyz_flat = np.asarray(dataset_file["inputs_t"][:, coord_feature_indices], dtype=np.float32)
        coord_min = np.min(xyz_flat, axis=0).astype(np.float32)
        coord_span = np.ptp(xyz_flat, axis=0).astype(np.float32)
    grid_resolution = (0, 0, 0)
else:
    feature_names_all = [str(x) for x in dataset_file["feature_names"].tolist()]
    target_names = [str(x) for x in dataset_file["target_names"].tolist()]
    requested_channels = _csv_env("FINAL2_GINO_INPUT_CHANNELS", DEFAULT_FIELD_CHANNELS)
    sample_paths = [resolve_sample_path(p, DATASET_PATH.parent) for p in dataset_file["sample_paths"].tolist()]
    train_frame_ids = split_ids(dataset_file, "train_frame_ids")
    val_id_frame_ids = split_ids(dataset_file, "val_id_frame_ids")
    val_angle_frame_ids = split_ids(dataset_file, "val_angle_frame_ids")
    test_normal_frame_ids = split_ids(dataset_file, "test_normal_frame_ids")
    test_spatial_sr_frame_ids = split_ids(dataset_file, "test_spatial_sr_frame_ids")
    test_temporal_sr_frame_ids = split_ids(dataset_file, "test_temporal_sr_frame_ids")
    test_unseen_angle_frame_ids = split_ids(dataset_file, "test_unseen_angle_frame_ids")
    coord_feature_indices = np.asarray([0, 1, 2], dtype=np.int64)
    coord_min = np.asarray(dataset_file["coord_min"], dtype=np.float32).reshape(3)
    coord_span = np.asarray(dataset_file["coord_span"], dtype=np.float32).reshape(3)
    grid_resolution = tuple(int(x) for x in dataset_file["grid_resolution"].tolist())

missing_channels = [name for name in requested_channels if name not in feature_names_all]
if missing_channels:
    raise KeyError(f"Requested input channels not in processed data: {missing_channels}. Available: {feature_names_all}")


# In[ ]:


active_input_feature_indices = [feature_names_all.index(name) for name in requested_channels]
feature_names = [feature_names_all[i] for i in active_input_feature_indices]
removed_input_features = [name for name in feature_names_all if name not in feature_names]

input_mean_all = np.asarray(dataset_file["in_mean"], dtype=np.float32).reshape(-1)
input_std_all = np.maximum(np.asarray(dataset_file["in_std"], dtype=np.float32).reshape(-1), 1e-8)
input_mean = input_mean_all[active_input_feature_indices]
input_std = input_std_all[active_input_feature_indices]
target_mean = np.asarray(dataset_file["out_mean"], dtype=np.float32).reshape(-1)
target_std = np.maximum(np.asarray(dataset_file["out_std"], dtype=np.float32).reshape(-1), 1e-8)
coord_span = np.maximum(np.asarray(coord_span, dtype=np.float32), 1e-8)


# In[ ]:


# print('Number of frames      :', len(inputs_by_frame_normalized))
# print('Input feature count   :', input_dimension)
# print('Output target count   :', output_dimension)
# print('Training frames       :', len(training_frame_ids))
# print('Validation frames     :', len(validation_frame_ids), '(same-distribution frames from training cases)')
# print('Held-out angle validation frames:', len(validation_angle_frame_ids))
# print('Testing frames        :', len(testing_frame_ids))
# print('  normal test frames  :', len(testing_normal_frame_ids))
# print('  super-res test frames:', len(testing_super_resolution_frame_ids))
# print('  unseen-angle frames :', len(testing_unseen_angle_frame_ids))
# print('Input feature names   :', feature_names)
# print('Freestream vector channels used:', use_freestream_vector_features_as_model_input)
# print('Removed input features:', removed_input_features if removed_input_features else 'none')
# print('Output target names   :', target_names)
# if not has_validation_data:
#     print('[info] No validation frames yet. The notebook will skip validation plots/metrics until those cases exist.')
# if not has_testing_data:
#     print('[info] No testing frames yet. The notebook will skip testing plots/metrics until those cases exist.')




# In[ ]:


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.clip((xyz.astype(np.float32) - coord_min[None, :]) / coord_span[None, :], 0.0, 1.0).astype(np.float32)


def make_dataset(frame_ids: np.ndarray, split_name: str, shuffle: bool, max_output_points: Optional[int]):
    if TASK == "particle_ugradu":
        ds = ParticleUGradUDataset(dataset_file, active_input_feature_indices, frame_ids, split_name, CFG["maximum_input_particles"], max_output_points)
    else:
        ds = FieldReconstructionDataset(sample_paths, active_input_feature_indices, frame_ids, split_name, CFG["maximum_input_particles"], max_output_points)
    loader = DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=CFG["num_workers"])
    return ds, loader


testing_frame_ids = test_normal_frame_ids
if len(testing_frame_ids) == 0:
    testing_frame_ids = val_angle_frame_ids

train_ds, train_loader = make_dataset(train_frame_ids, "train", True, CFG["maximum_train_output_points"])
val_id_ds, val_id_loader = make_dataset(val_id_frame_ids, "val_id", False, CFG["maximum_eval_output_points"])
val_angle_ds, val_angle_loader = make_dataset(val_angle_frame_ids, "val_angle", False, CFG["maximum_eval_output_points"])
test_ds, test_loader = make_dataset(testing_frame_ids, "test", False, CFG["maximum_eval_output_points"])

print("=" * 72)
print("Task        :", TASK)
print("Field decoder:", FIELD_DECODER if TASK == "field_reconstruction" else "n/a")
print("Dataset     :", DATASET_PATH)
print("Results dir :", RESULTS_DIR)
print("Device      :", DEVICE)
print("Input names :", feature_names)
print("Removed     :", removed_input_features if removed_input_features else "none")
print("Targets     :", target_names)
print("Splits      :", {k: len(v) for k, v in {
    "train": train_frame_ids,
    "val_id": val_id_frame_ids,
    "val_angle": val_angle_frame_ids,
    "test_normal": test_normal_frame_ids,
    "test_spatial_sr": test_spatial_sr_frame_ids,
    "test_temporal_sr": test_temporal_sr_frame_ids,
    "test_unseen": test_unseen_angle_frame_ids,
}.items()})
print("Config      :", json.dumps(CFG, indent=2))
print("=" * 72)


# In[ ]:


# Loader sanity check: catches stale dataset definitions before the long training loop.
_probe_batch = next(iter(train_loader))
print("loader sanity input_geom", tuple(_probe_batch["input_geom"].shape))
print("loader sanity x", tuple(_probe_batch["x"].shape))
print("loader sanity output_queries", tuple(_probe_batch["output_queries"].shape))
print("loader sanity y", tuple(_probe_batch["y"].shape))


# In[ ]:


def make_latent_queries(res: int, device: torch.device) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, int(res), dtype=torch.float32, device=device)
    xx, yy, zz = torch.meshgrid(line, line, line, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)


LATENT_QUERIES = make_latent_queries(CFG["latent_res"], DEVICE)


# In[ ]:


def build_neuralop_gino(cfg: Dict) -> nn.Module:
    if GINO is None:
        raise RuntimeError("neuralop.models.GINO is not available") from GINO_IMPORT_ERROR
    kwargs = dict(
        in_channels=len(feature_names),
        out_channels=len(target_names),
        gno_coord_dim=3,
        in_gno_radius=cfg["in_gno_radius"],
        out_gno_radius=cfg["out_gno_radius"],
        in_gno_transform_type=cfg["in_gno_transform_type"],
        out_gno_transform_type=cfg["out_gno_transform_type"],
        gno_embed_channels=cfg["gno_embed_channels"],
        gno_use_open3d=cfg["gno_use_open3d"],
        gno_use_torch_scatter=cfg["gno_use_torch_scatter"],
        fno_n_modes=tuple(cfg["fno_n_modes"]),
        fno_hidden_channels=cfg["fno_hidden_channels"],
        fno_n_layers=cfg["fno_n_layers"],
        projection_channel_ratio=cfg["projection_channel_ratio"],
    )
    accepted = set(inspect.signature(GINO).parameters)
    return GINO(**{k: v for k, v in kwargs.items() if k in accepted}).to(DEVICE)



# In[ ]:


class PointwiseLatentGINO(nn.Module):
    """GNO encoder + latent grid + trilinear pointwise decoder.

    This is the toggleable field option where the expensive output GNO decoder
    is replaced by a pointwise decoder at arbitrary query coordinates.
    """

    def __init__(self, in_channels: int, out_channels: int, cfg: Dict):
        super().__init__()
        if GNOBlock is None:
            raise RuntimeError("neuralop.layers.gno_block.GNOBlock is not available") from GNOBLOCK_IMPORT_ERROR
        hidden = int(cfg["fno_hidden_channels"])
        self.latent_res = int(cfg["latent_res"])
        self.lift = nn.Sequential(nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.in_gno = GNOBlock(
            in_channels=hidden,
            out_channels=hidden,
            coord_dim=3,
            radius=float(cfg["in_gno_radius"]),
            transform_type=str(cfg["in_gno_transform_type"]),
            reduction="mean",
            pos_embedding_type="transformer",
            pos_embedding_channels=12,
            channel_mlp_layers=[hidden, hidden, hidden],
            use_torch_scatter_reduce=bool(cfg["gno_use_torch_scatter"]),
            use_open3d_neighbor_search=bool(cfg["gno_use_open3d"]),
        )
        layers = []
        for _ in range(max(int(cfg["fno_n_layers"]), 1)):
            layers += [nn.Conv3d(hidden, hidden, kernel_size=3, padding=1), nn.GELU()]
        self.latent_mixer = nn.Sequential(*layers)
        mlp_layers: List[nn.Module] = []
        width = int(cfg["pointwise_hidden"])
        in_dim = hidden + 3
        for i in range(max(int(cfg["pointwise_layers"]), 1)):
            mlp_layers.append(nn.Linear(in_dim if i == 0 else width, width))
            mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(width, out_channels))
        self.decoder = nn.Sequential(*mlp_layers)

    def forward(self, input_geom: torch.Tensor, latent_queries: torch.Tensor, output_queries: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bsz = int(x.shape[0])
        outs = []
        base_latent = latent_queries[0]
        for b in range(bsz):
            h = self.lift(x[b])
            latent = self.in_gno(y=base_latent, x=input_geom[b], f_y=h)
            if latent.ndim == 3:
                latent = latent.squeeze(0)
            r = self.latent_res
            grid = latent.reshape(r, r, r, -1).permute(3, 0, 1, 2).unsqueeze(0)
            grid = self.latent_mixer(grid)
            q = output_queries[b].clamp(0.0, 1.0)
            sample_grid = (q * 2.0 - 1.0).view(1, -1, 1, 1, 3)
            sampled = torch.nn.functional.grid_sample(grid, sample_grid, align_corners=True, mode="bilinear")
            sampled = sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)
            outs.append(self.decoder(torch.cat([sampled, q], dim=-1)))
        return torch.stack(outs, dim=0)


# In[ ]:


def build_model(cfg: Dict) -> nn.Module:
    if TASK == "field_reconstruction" and FIELD_DECODER == "pointwise":
        return PointwiseLatentGINO(len(feature_names), len(target_names), cfg).to(DEVICE)
    return build_neuralop_gino(cfg)


MODEL = build_model(CFG)
TOTAL_PARAMS = count_model_params(MODEL)
TRAINABLE_PARAMS = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
print("Latent queries:", tuple(LATENT_QUERIES.shape))
print("Parameters    :", f"{TOTAL_PARAMS:,}", "total /", f"{TRAINABLE_PARAMS:,}", "trainable")

target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
target_std_t = torch.tensor(target_std, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
target_var_t = torch.tensor(target_std**2, dtype=torch.float32, device=DEVICE).view(1, 1, -1).clamp_min(1e-12)


# In[ ]:


def move_batch(batch: Dict) -> Dict:
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model: nn.Module, batch: Dict) -> torch.Tensor:
    return model(
        input_geom=batch["input_geom"],
        latent_queries=LATENT_QUERIES,
        output_queries=batch["output_queries"],
        x=batch["x"],
    )


def denormalize_target(y_norm: torch.Tensor) -> torch.Tensor:
    return y_norm * target_std_t + target_mean_t


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diff = (pred - target).reshape(pred.shape[0], -1)
    ref = target.reshape(target.shape[0], -1)
    return torch.linalg.norm(diff, dim=1) / torch.linalg.norm(ref, dim=1).clamp_min(eps)


# def grouped_training_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     pred_phys = denormalize_target(pred)
#     target_phys = denormalize_target(target)
#     mse_per_channel = torch.mean((pred_phys - target_phys) ** 2, dim=(0, 1))
#     scaled = mse_per_channel / target_var_t.reshape(-1)
#     if len(target_names) == 12:
#         primary = torch.mean(scaled[:3])
#         secondary = torch.mean(scaled[3:])
#     elif len(target_names) >= 6:
#         primary = torch.mean(scaled[:3])
#         secondary = torch.mean(scaled[3:6])
#     else:
#         primary = torch.mean(scaled)
#         secondary = torch.zeros((), dtype=primary.dtype, device=primary.device)
#     return primary + secondary, primary.detach(), secondary.detach()


def grouped_training_loss(pred, target):
    pred_phys = denormalize_target(pred)
    target_phys = denormalize_target(target)
    mse = torch.mean((pred_phys - target_phys) ** 2)
    # Optional: separate losses for logging velocity / gradient
    vel_mse = torch.mean((pred_phys[:, :3] - target_phys[:, :3]) ** 2)
    grad_mse = torch.mean((pred_phys[:, 3:] - target_phys[:, 3:]) ** 2)
    return mse, vel_mse.detach(), grad_mse.detach()


# In[ ]:


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, collect: bool = False) -> Dict:
    model.eval()
    if len(loader.dataset) == 0:
        return {"loss": math.nan, "rel_l2_norm": math.nan, "rel_l2_phys": math.nan, "rmse_phys": math.nan, "mae_phys": math.nan, "rel_per_sample": np.asarray([], dtype=np.float32)}
    sums = defaultdict(float)
    rels = []
    n = 0
    for batch in loader:
        batch = move_batch(batch)
        pred = predict(model, batch)
        y = batch["y"]
        loss, _, _ = grouped_training_loss(pred, y)
        pred_phys = denormalize_target(pred)
        y_phys = denormalize_target(y)
        sums["loss"] += float(loss.item())
        sums["rel_l2_norm"] += float(relative_l2(pred, y).mean().item())
        sums["rel_l2_phys"] += float(relative_l2(pred_phys, y_phys).mean().item())
        sums["rmse_phys"] += float(torch.sqrt(torch.mean((pred_phys - y_phys) ** 2)).item())
        sums["mae_phys"] += float(torch.mean(torch.abs(pred_phys - y_phys)).item())
        if collect:
            rels.extend(relative_l2(pred_phys, y_phys).detach().cpu().numpy().tolist())
        n += 1
    out = {k: v / max(n, 1) for k, v in sums.items()}
    if collect:
        out["rel_per_sample"] = np.asarray(rels, dtype=np.float32)
    return out


# In[ ]:


history = {
    "epoch": [],
    "train_loss": [],
    "train_rel_l2_norm": [],
    "train_primary_loss": [],
    "train_secondary_loss": [],
    "val_id_rel_l2_phys": [],
    "val_angle_rel_l2_phys": [],
    "test_rel_l2_phys": [],
    "test_rmse_phys": [],
    "lr": [],
    "bad_batches": [],
    "optimizer_steps": [],
    "train_batches": [],
}


# In[ ]:


optimizer = torch.optim.AdamW(MODEL.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(CFG["epochs"]), 1))
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and CFG["use_amp"]))
except TypeError:  # older torch
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda" and CFG["use_amp"]))
autocast_options = dict(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda" and CFG["use_amp"]))


# In[ ]:


def checkpoint_payload(tag: str, score: float, extra_metrics: Optional[Dict] = None) -> Dict:
    payload = {
        "checkpoint_tag": tag,
        "saved_at_utc": datetime.utcnow().isoformat() + "Z",
        "config": dict(CFG),
        "model_state_dict": MODEL.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "feature_names_all": feature_names_all,
        "feature_names": feature_names,
        "target_names": target_names,
        "active_input_feature_indices": active_input_feature_indices,
        "removed_input_features": removed_input_features,
        "dataset_path": portable_path(DATASET_PATH, FINAL_DIR),
        "results_dir": portable_path(RESULTS_DIR, FINAL_DIR),
        "grid_resolution": grid_resolution,
        "coord_min": coord_min,
        "coord_span": coord_span,
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_frame_ids": train_frame_ids,
        "val_id_frame_ids": val_id_frame_ids,
        "val_angle_frame_ids": val_angle_frame_ids,
        "test_normal_frame_ids": test_normal_frame_ids,
        "test_spatial_sr_frame_ids": test_spatial_sr_frame_ids,
        "test_temporal_sr_frame_ids": test_temporal_sr_frame_ids,
        "test_unseen_angle_frame_ids": test_unseen_angle_frame_ids,
        "best_score": float(score),
        "history": history,
        "model_param_count": int(TOTAL_PARAMS),
        "trainable_param_count": int(TRAINABLE_PARAMS),
    }
    if extra_metrics is not None:
        payload["metrics"] = extra_metrics
    return payload


def save_checkpoint(path: Path, tag: str, score: float, extra_metrics: Optional[Dict] = None) -> None:
    torch.save(checkpoint_payload(tag, score, extra_metrics), path)
    print(f"[saved checkpoint] {path}")


def save_history() -> None:
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"[saved history] {HISTORY_PATH}")


if len(train_loader.dataset) == 0:
    raise RuntimeError("Training split is empty.")

best_score = float("inf")


# In[ ]:


for epoch in range(1, int(CFG["epochs"]) + 1):
    MODEL.train()
    start_time = time.time()
    accumulation_steps = max(int(CFG["gradient_accumulation_steps"]), 1)
    optimizer.zero_grad(set_to_none=True)
    train_losses, train_rels, primary_losses, secondary_losses = [], [], [], []
    bad_batches = 0
    optimizer_steps = 0
    good_batches = 0

    for local_step, batch in enumerate(train_loader, start=1):
        batch = move_batch(batch)
        with torch.autocast(**autocast_options):
            pred = predict(MODEL, batch)
            loss, primary_loss, secondary_loss = grouped_training_loss(pred, batch["y"])
            loss_for_backward = loss / accumulation_steps
        if not torch.isfinite(loss):
            bad_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        should_step = local_step % accumulation_steps == 0 or local_step == len(train_loader)
        if should_step:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if CFG["grad_clip_norm"] is not None and CFG["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(MODEL.parameters(), CFG["grad_clip_norm"])
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        with torch.no_grad():
            train_losses.append(float(loss.item()))
            train_rels.append(float(relative_l2(pred, batch["y"]).mean().item()))
            primary_losses.append(float(primary_loss.item()))
            secondary_losses.append(float(secondary_loss.item()))
            good_batches += 1

    scheduler.step()
    if good_batches == 0:
        raise RuntimeError("All training batches were non-finite.")

    if epoch % int(CFG["eval_every"]) == 0 or epoch == int(CFG["epochs"]):
        val_id_metrics = evaluate(MODEL, val_id_loader)
        val_angle_metrics = evaluate(MODEL, val_angle_loader)
        test_metrics = evaluate(MODEL, test_loader)
        train_loss = float(np.mean(train_losses))
        train_rel = float(np.mean(train_rels))
        score = val_id_metrics["rel_l2_phys"]
        if not np.isfinite(score):
            score = val_angle_metrics["rel_l2_phys"] if np.isfinite(val_angle_metrics["rel_l2_phys"]) else test_metrics["rel_l2_phys"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_rel_l2_norm"].append(train_rel)
        history["train_primary_loss"].append(float(np.mean(primary_losses)))
        history["train_secondary_loss"].append(float(np.mean(secondary_losses)))
        history["val_id_rel_l2_phys"].append(val_id_metrics["rel_l2_phys"])
        history["val_angle_rel_l2_phys"].append(val_angle_metrics["rel_l2_phys"])
        history["test_rel_l2_phys"].append(test_metrics["rel_l2_phys"])
        history["test_rmse_phys"].append(test_metrics["rmse_phys"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["bad_batches"].append(int(bad_batches))
        history["optimizer_steps"].append(int(optimizer_steps))
        history["train_batches"].append(int(good_batches))
        metrics_bundle = {
            "train_loss": train_loss,
            "train_rel_l2_norm": train_rel,
            "val_id_rel_l2_phys": val_id_metrics["rel_l2_phys"],
            "val_angle_rel_l2_phys": val_angle_metrics["rel_l2_phys"],
            "test_rel_l2_phys": test_metrics["rel_l2_phys"],
            "test_rmse_phys": test_metrics["rmse_phys"],
        }
        save_checkpoint(LAST_CKPT_PATH, "last", score, metrics_bundle)
        save_history()
        if np.isfinite(score) and score < best_score:
            best_score = float(score)
            save_checkpoint(CKPT_PATH, "best", best_score, metrics_bundle)
        elapsed = time.time() - start_time
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4e} train_rel={train_rel:.4e} "
            f"val_id={val_id_metrics['rel_l2_phys']:.4e} val_angle={val_angle_metrics['rel_l2_phys']:.4e} "
            f"test={test_metrics['rel_l2_phys']:.4e} rmse={test_metrics['rmse_phys']:.4e} "
            f"good={good_batches} bad={bad_batches} steps={optimizer_steps} "
            f"lr={history['lr'][-1]:.2e} time={elapsed:.1f}s"
        )


# In[ ]:


if not CKPT_PATH.exists():
    fallback_score = history["test_rel_l2_phys"][-1] if history["test_rel_l2_phys"] else float("inf")
    save_checkpoint(CKPT_PATH, "best_fallback", fallback_score)

print("Best score     :", best_score)
print("Best checkpoint:", CKPT_PATH)
print("Last checkpoint:", LAST_CKPT_PATH)
print("History        :", HISTORY_PATH)

