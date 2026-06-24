#!/usr/bin/env python
# coding: utf-8


# %% [markdown]
# # Updated GINO model - 23 June 2026
# 
# Targets - $u_x, u_y, u_z$, $\nabla{u}_x, \nabla{u}_y, \nabla{u}_z$
# 
# 
# From the predicted particle velocity - use either sr, or a pointwise decoder from latent space for reconstructing flow field. Use the predicted values to propogate the states

# %% [markdown]
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

# %% cell 2
from __future__ import annotations
import socket

print(socket.gethostname())

# %% [markdown]
# Include processing and visualizing data snippets from final-2/task1 notebook.
# Check if all the required parameters are saved in the .pt file and required plots should be put in a new file, including loading the trained model. 

# %% cell 4
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
import torch.nn.functional as F
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
    from neuralop.layers.neighbor_search import NeighborSearch
    NEIGHBOR_SEARCH_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - depends on runtime env
    NeighborSearch = None
    NEIGHBOR_SEARCH_IMPORT_ERROR = error

try:
    from neuralop.utils import count_model_params
except Exception:  # pragma: no cover
    def count_model_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

print('Torch       :', torch.__version__)
print('CUDA build  :', torch.version.cuda)
print('CUDA usable :', torch.cuda.is_available())

# %% cell 5
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
if FIELD_DECODER not in {"pointwise","super_resolution"}:
    raise ValueError("GINO_FIELD_DECODER must be super_resolution or pointwise")

print(TASK,FIELD_DECODER,REPO_ROOT, FINAL_DIR, DEVICE)

# %% cell 6
DEFAULT_PARTICLE_CHANNELS = [
    "Gamma_x",
    "Gamma_y",
    "Gamma_z",
    "sigma",
    "geom_dist",
    "geom_nx",
    "geom_ny",
    "geom_nz",
    "geom_body_near",
    "angle_of_attack",
    "freestream_x",
    "freestream_z",
    "phase",
]
DEFAULT_FIELD_CHANNELS = DEFAULT_PARTICLE_CHANNELS

# %% cell 7
CFG = {
    # ===== Basic =====
    "seed": SEED,
    "task": os.environ.get("GINO_TASK", "particle_ugradu").strip().lower(),
    "field_decoder": os.environ.get("GINO_FIELD_DECODER", "super_resolution").strip().lower(),
    # Task 1 defaults to the vortex-inspired edge GNO because the current NeuralOperator
    # GINO run is unstable for particle U/gradU. Set GINO_MODEL_FAMILY=gino for ablations
    # or for the field-reconstruction task.
    "model_family": os.environ.get("GINO_MODEL_FAMILY", "edge_gno").strip().lower(),
    "file_tag": os.environ.get("GINO_FILE_TAG", "task1_particle_ugradu_edge_gno"),

    # ===== Training =====
    "epochs": int(os.environ.get("GINO_EPOCHS", "60")),
    "lr": float(os.environ.get("GINO_LR", "3e-4")),
    "min_lr": float(os.environ.get("GINO_MIN_LR", "5e-6")),
    "weight_decay": float(os.environ.get("GINO_WEIGHT_DECAY", "3e-5")),
    "eval_every": int(os.environ.get("GINO_EVAL_EVERY", "2")),
    "gradient_accumulation_steps": int(os.environ.get("GINO_ACCUM_STEPS", "4")),
    "grad_clip_norm": float(os.environ.get("GINO_GRAD_CLIP", "1.0")),
    "batch_size": 1,
    "num_workers": int(os.environ.get("GINO_NUM_WORKERS", "0")),

    # ===== Particle curriculum / evaluation caps =====
    "particles_per_frame_start": int(os.environ.get("GINO_PARTICLES_START", "1024")),
    "particles_per_frame_final": int(os.environ.get("GINO_PARTICLES_FINAL", "4096")),
    "maximum_input_particles": int(os.environ.get("GINO_MAX_INPUT_PARTICLES", "4096")),
    "maximum_train_output_points": int(os.environ.get("GINO_MAX_TRAIN_OUTPUT_POINTS", "4096")),
    "maximum_eval_output_points": int(os.environ.get("GINO_MAX_EVAL_OUTPUT_POINTS", "4096")),
    "eval_max_frames": int(os.environ.get("GINO_EVAL_MAX_FRAMES", "16")),
    "minimum_target_norm_for_relative_error": float(os.environ.get("GINO_MIN_REL_TARGET_NORM", "1e-6")),
    "minimum_target_norm_to_keep": float(os.environ.get("GINO_MIN_KEEP_TARGET_NORM", "1e-12")),

    # ===== Loss =====
    "loss_balance_mode": os.environ.get("GINO_LOSS_BALANCE", "variance_normalized_physical_mse"),
    "loss_name": os.environ.get("GINO_LOSS_NAME", "mse"),
    "smooth_l1_beta": float(os.environ.get("GINO_SMOOTH_L1_BETA", "0.06")),

    # ===== Mixed precision =====
    "use_amp": _bool_env("GINO_USE_AMP", True),

    # ===== Vortex-inspired edge GNO for Task 1 =====
    "edge_hidden_size": int(os.environ.get("GINO_EDGE_HIDDEN", "128")),
    "edge_graph_layers": int(os.environ.get("GINO_EDGE_LAYERS", "3")),
    "edge_neighbor_radius": float(os.environ.get("GINO_EDGE_RADIUS", "0.12")),
    "edge_dropout": float(os.environ.get("GINO_EDGE_DROPOUT", "0.06")),
    "edge_use_physical_features": _bool_env("GINO_EDGE_PHYSICAL_FEATURES", True),

    # ===== NeuralOperator GINO / latent field path =====
    "latent_res": int(os.environ.get("GINO_LATENT_RES", "12")),
    "in_gno_radius": float(os.environ.get("GINO_IN_RADIUS", "0.10")),
    "out_gno_radius": float(os.environ.get("GINO_OUT_RADIUS", "0.12")),
    "in_gno_transform_type": os.environ.get("GINO_IN_TRANSFORM", "nonlinear_kernelonly"),
    "out_gno_transform_type": os.environ.get("GINO_OUT_TRANSFORM", "linear"),
    "gno_embed_channels": int(os.environ.get("GINO_EMBED_CHANNELS", "24")),
    "fno_n_modes": tuple(int(x) for x in os.environ.get("GINO_FNO_MODES", "4,4,4").split(",")),
    "fno_hidden_channels": int(os.environ.get("GINO_FNO_HIDDEN", "24")),
    "fno_n_layers": int(os.environ.get("GINO_FNO_LAYERS", "4")),
    "projection_channel_ratio": int(os.environ.get("GINO_PROJ_RATIO", "2")),
    "gno_use_open3d": _bool_env("GINO_USE_OPEN3D", False),
    "gno_use_torch_scatter": _bool_env("GINO_USE_TORCH_SCATTER", False),
    "pointwise_hidden": int(os.environ.get("GINO_POINTWISE_HIDDEN", "96")),
    "pointwise_layers": int(os.environ.get("GINO_POINTWISE_LAYERS", "3")),
}

TASK = CFG["task"]
FIELD_DECODER = CFG["field_decoder"]
if TASK not in {"particle_ugradu", "field_reconstruction"}:
    raise ValueError("GINO_TASK must be particle_ugradu or field_reconstruction")
if FIELD_DECODER not in {"super_resolution", "pointwise"}:
    raise ValueError("GINO_FIELD_DECODER must be super_resolution or pointwise")
if CFG["model_family"] not in {"edge_gno", "gino"}:
    raise ValueError("GINO_MODEL_FAMILY must be edge_gno or gino")
if TASK != "particle_ugradu" and CFG["model_family"] == "edge_gno":
    print("[info] edge_gno is particle-only; switching field task to model_family='gino'.")
    CFG["model_family"] = "gino"

# %% cell 8
def _find_task2_dataset() -> Path:
    env = os.environ.get("BACKUP_TASK_DATASET", "").strip()
    candidates = [Path(env)] if env else []
    candidates += [
        FINAL_DIR / "processed_data" / "Backup Data" / "task2_fdom_dataset.npz",
        REPO_ROOT / "final-2" / "processed_data_task2" / "task2_gino_dataset.npz",
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
        FINAL_DIR / "processed_data" / "task1" / "particle_ugradu_dataset.npz",
        REPO_ROOT / "final-2" / "processed_data_task1" / "particle_ugradu_dataset.npz",
        Path("/media/dysco/New Volume1/Neeraj/neuralop/processed_data/particle_ugradu_dataset.npz"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find particle_ugradu_dataset.npz.")


DATASET_PATH = _find_particle_dataset() if TASK == "particle_ugradu" else _find_task2_dataset()
RESULTS_DIR = FINAL_DIR / "result" / ("task1" if TASK == "particle_ugradu" else "task2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = RESULTS_DIR / f"{CFG['file_tag']}_best_model.pt"
LAST_CKPT_PATH = RESULTS_DIR / f"{CFG['file_tag']}_last_model.pt"
HISTORY_PATH = RESULTS_DIR / f"{CFG['file_tag']}_history.json"


print("=" * 60)
print("DATASET_PATH      :", DATASET_PATH)
print("RESULTS_DIR       :", RESULTS_DIR)
print("CKPT_PATH         :", CKPT_PATH)
print("LAST_CKPT_PATH    :", LAST_CKPT_PATH)
print("HISTORY_PATH      :", HISTORY_PATH)
print("FINAL_DIR         :", FINAL_DIR)
print("TASK              :", TASK)
print("FIELD_DECODER     :", FIELD_DECODER)
print("=" * 60)

# %% cell 9
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

# %% cell 10
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
            "input_geom_physical": torch.from_numpy(input_features[:, coord_feature_indices].astype(np.float32)),
            "x": torch.from_numpy(np.clip(np.nan_to_num(x), -8.0, 8.0).astype(np.float32)),
            "output_queries": torch.from_numpy(output_queries),
            "output_indices": torch.from_numpy(out_idx.astype(np.int64)),
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
            "input_geom_physical": torch.from_numpy(input_geom_raw[in_idx].astype(np.float32)),
            "x": torch.from_numpy(np.clip(np.nan_to_num(x), -8.0, 8.0).astype(np.float32)),
            "output_queries": torch.from_numpy(normalize_xyz(queries_raw[out_idx])),
            "y": torch.from_numpy(np.nan_to_num(y).astype(np.float32)),
            "frame_id": torch.tensor(frame_id, dtype=torch.long),
        }

# %% cell 11
def resolve_sample_path(path_like, base: Path) -> Path:
    raw = Path(str(path_like))
    candidates = [raw] if raw.is_absolute() else []
    candidates += [base / raw, base / "task2_gino_frames" / raw.name, FINAL_DIR / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing sample path {raw}; checked {candidates}")


dataset_file = _safe_np_load(DATASET_PATH)

# %% cell 12
if TASK == "particle_ugradu":

    feature_names_all = [str(x) for x in dataset_file["feature_names"].tolist()]
    target_names = [str(x) for x in dataset_file["target_names"].tolist()]

    requested_channels = _csv_env("GINO_INPUT_CHANNELS", DEFAULT_PARTICLE_CHANNELS)

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
    requested_channels = _csv_env("GINO_INPUT_CHANNELS", DEFAULT_FIELD_CHANNELS)
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
    env_channels = os.environ.get("GINO_INPUT_CHANNELS", "").strip()
    if env_channels:
        raise KeyError(f"Requested input channels not in processed data: {missing_channels}. Available: {feature_names_all}")
    print(f"[warn] Default channels missing from this dataset and will be skipped: {missing_channels}")
    requested_channels = [name for name in requested_channels if name in feature_names_all]

# %% cell 13
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

# %% [markdown]
# Snippet to see the input channels and the output channels

# %% cell 15
print('All stored input features:')
print(feature_names_all)
print()
print('Model input features after channel selection:')
print(feature_names)
print()
print('Removed input features:')
print(removed_input_features if removed_input_features else 'none')
print()
print('Output target names:')
print(target_names)
print()

# %% cell 16
def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.clip((xyz.astype(np.float32) - coord_min[None, :]) / coord_span[None, :], 0.0, 1.0).astype(np.float32)


def target_norm_for_frame(frame_id: int) -> float:
    target = np.asarray(dataset_file["targets_by_frame"][int(frame_id)], dtype=np.float32)
    return float(np.linalg.norm(np.nan_to_num(target).reshape(-1)))


def filter_zero_target_frames(frame_ids: np.ndarray, split_name: str, minimum_norm: float) -> np.ndarray:
    kept, removed = [], []
    for frame_id in np.asarray(frame_ids, dtype=np.int64):
        norm_value = target_norm_for_frame(int(frame_id))
        if np.isfinite(norm_value) and norm_value > minimum_norm:
            kept.append(int(frame_id))
        else:
            context = as_context(dataset_file["frame_contexts"][int(frame_id)]) if "frame_contexts" in dataset_file.files else {}
            removed.append((int(frame_id), context.get("case", "unknown"), context.get("frame", "unknown"), norm_value))
    if removed:
        print(f"[filter] {split_name}: removed {len(removed)} tiny/zero-target frames; examples={removed[:3]}")
    return np.asarray(kept, dtype=np.int64)


minimum_target_norm_to_keep = float(CFG["minimum_target_norm_to_keep"])
if TASK == "particle_ugradu":
    train_frame_ids = filter_zero_target_frames(train_frame_ids, "train", minimum_target_norm_to_keep)
    val_id_frame_ids = filter_zero_target_frames(val_id_frame_ids, "val_id", minimum_target_norm_to_keep)
    val_angle_frame_ids = filter_zero_target_frames(val_angle_frame_ids, "val_angle", minimum_target_norm_to_keep)
    test_normal_frame_ids = filter_zero_target_frames(test_normal_frame_ids, "test_normal", minimum_target_norm_to_keep)
    test_spatial_sr_frame_ids = filter_zero_target_frames(test_spatial_sr_frame_ids, "test_spatial_sr", minimum_target_norm_to_keep)
    test_unseen_angle_frame_ids = filter_zero_target_frames(test_unseen_angle_frame_ids, "test_unseen_angle", minimum_target_norm_to_keep)


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
print("Model family:", CFG["model_family"])
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

# %% [markdown]
# should be run in the preprocessing script as removal occurs in that file

# %% cell 18
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------
# 1. Helper: get case name and per‑case frame number from a frame_id
# ------------------------------------------------------------
def get_case_and_frame(frame_id: int):
    # Access frame_contexts from the dataset file
    contexts = list(dataset_file["frame_contexts"])
    context = as_context(contexts[frame_id])
    case = context.get("case", context.get("case_name", "unknown"))
    # Per‑case frame number (may be stored as 'frame' or 'fr')
    frame_num = int(context.get("frame", context.get("fr", 0)))
    return case, frame_num

# ------------------------------------------------------------
# 2. Collect all original frame IDs from all splits (before filtering)
# ------------------------------------------------------------
all_original_ids = np.concatenate([
    split_ids(dataset_file, "train_frame_ids"),
    split_ids(dataset_file, "val_id_frame_ids"),
    split_ids(dataset_file, "validation_angle_frame_ids"),
    split_ids(dataset_file, "test_normal_frame_ids"),
    split_ids(dataset_file, "test_super_resolution_frame_ids"),
    split_ids(dataset_file, "test_unseen_angle_frame_ids"),
])
# Remove duplicates and sort
all_original_ids = np.unique(all_original_ids)

# Current kept frames (after zero‑target filtering) – these are your current split arrays
kept_ids = set(train_frame_ids.tolist() + val_id_frame_ids.tolist() +
               val_angle_frame_ids.tolist() + test_normal_frame_ids.tolist() +
               test_spatial_sr_frame_ids.tolist() + test_unseen_angle_frame_ids.tolist())

# Removed frames = all_original_ids - kept_ids
removed_ids = sorted(set(all_original_ids) - kept_ids)

# ------------------------------------------------------------
# 3. Group kept/removed per case
# ------------------------------------------------------------
case_kept = defaultdict(list)
case_removed = defaultdict(list)

for fid in all_original_ids:
    case, fnum = get_case_and_frame(fid)
    if fid in kept_ids:
        case_kept[case].append(fnum)
    else:
        case_removed[case].append((fnum, fid))

# Sort per‑case lists
for case in case_kept:
    case_kept[case].sort()
for case in case_removed:
    case_removed[case].sort(key=lambda x: x[0])

# ------------------------------------------------------------
# 4. Print statistics and examples
# ------------------------------------------------------------
print("=" * 80)
print("DIAGNOSTIC: Frame filtering analysis")
print("=" * 80)

print(f"Total original frames (all splits): {len(all_original_ids)}")
print(f"Total kept frames (after filtering): {len(kept_ids)}")
print(f"Total removed frames: {len(removed_ids)}")
print()

print("Per‑case kept frame counts (expected ~200 per case, minus early transients):")
for case, frames in sorted(case_kept.items()):
    print(f"  {case}: {len(frames)} frames kept")

print("\nPer‑case removed frame count and examples:")
for case, removed in sorted(case_removed.items()):
    print(f"  {case}: {len(removed)} frames removed")
    if removed:
        # Show first few removed frames with their per‑case frame number and particle count / target norm
        examples = removed[:5]
        for fnum, fid in examples:
            # Get particle count and target norm for this frame
            try:
                features = np.asarray(dataset_file["inputs_by_frame"][fid], dtype=np.float32)
                targets = np.asarray(dataset_file["targets_by_frame"][fid], dtype=np.float32)
                n_particles = features.shape[0]
                target_norm = np.linalg.norm(targets.reshape(-1))
                # Get context to see the physical frame number
                context = as_context(list(dataset_file["frame_contexts"])[fid])
                phys_frame = context.get("frame", context.get("fr", "?"))
                print(f"      frame {fnum} (global {fid}, phys {phys_frame}) | particles={n_particles}, target_norm={target_norm:.2e}")
            except Exception as e:
                print(f"      frame {fnum} (global {fid}) -> error retrieving data: {e}")
        if len(removed) > 5:
            print(f"      ... and {len(removed)-5} more removed frames.")

# ------------------------------------------------------------
# 5. Count zero‑particle and zero‑target removed frames
# ------------------------------------------------------------
print("\nDetailed check for removed frames that might be startup / zero‑target:")
zero_particle = []
zero_target = []
for fid in removed_ids:
    try:
        features = np.asarray(dataset_file["inputs_by_frame"][fid], dtype=np.float32)
        targets = np.asarray(dataset_file["targets_by_frame"][fid], dtype=np.float32)
        n_particles = features.shape[0]
        target_norm = np.linalg.norm(targets.reshape(-1))
        if n_particles == 0:
            zero_particle.append(fid)
        if target_norm < 1e-12:
            zero_target.append(fid)
    except:
        pass

print(f"  Removed frames with zero particles: {len(zero_particle)}")
print(f"  Removed frames with zero target norm: {len(zero_target)}")
if zero_target:
    print("    Examples (global frame IDs):", zero_target[:10])

# %% [markdown]
# Reasons for mismatching frame ids and frame reset ids
# - Start-up frames
# - No particle frames
# - Zero target frames (early transients)
# - Validation and test case removal?

# %% cell 20
sample = train_ds[100]
print("input_geom shape:", sample["input_geom"].shape)
print("input_geom min/max:", sample["input_geom"].min().item(), sample["input_geom"].max().item())
print("x shape:", sample["x"].shape)
print("output_queries shape:", sample["output_queries"].shape)
print("y shape:", sample["y"].shape)
print("frame_id:", sample["frame_id"].item())

# %% [markdown]
# Output queries - matrix of dimension N (particles) with spatial coordinates of each particle. Locations where the y (N,12) is queried
# 
# Input_geom - (N,3) of spatial coordinates of particles. See where is it inputted to the model instead of the input channel?

# %% cell 22
# Loader sanity check: catches stale dataset definitions before the long training loop.
_probe_batch = next(iter(train_loader))
print("loader sanity input_geom", tuple(_probe_batch["input_geom"].shape))
print("loader sanity x", tuple(_probe_batch["x"].shape))
print("loader sanity output_queries", tuple(_probe_batch["output_queries"].shape))
print("loader sanity y", tuple(_probe_batch["y"].shape))

# %% cell 23
def make_latent_queries(res: int, device: torch.device) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, int(res), dtype=torch.float32, device=device)
    xx, yy, zz = torch.meshgrid(line, line, line, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)


LATENT_QUERIES = make_latent_queries(CFG["latent_res"], DEVICE)

# %% cell 24
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


# %% cell 25
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


class EdgeFeatureMessageBlock(nn.Module):
    """Radius-graph block with vortex-inspired pairwise features."""

    def __init__(self, hidden_size: int, neighbor_radius: float, use_open3d_neighbor_search: bool = False, dropout: float = 0.04):
        super().__init__()
        if NeighborSearch is None:
            raise RuntimeError("neuralop.layers.neighbor_search.NeighborSearch is not available") from NEIGHBOR_SEARCH_IMPORT_ERROR
        self.neighbor_radius = float(neighbor_radius)
        self.neighbor_search = NeighborSearch(use_open3d=bool(use_open3d_neighbor_search), return_norm=False)
        edge_feature_size = 2 * int(hidden_size) + 14
        self.message_network = nn.Sequential(
            nn.Linear(edge_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.normalization = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, search_positions: torch.Tensor, edge_positions: torch.Tensor, gamma_values: torch.Tensor, sigma_values: torch.Tensor) -> torch.Tensor:
        neighbors = self.neighbor_search(data=search_positions, queries=search_positions, radius=self.neighbor_radius)
        source_index = neighbors["neighbors_index"].long()
        row_splits = neighbors["neighbors_row_splits"].long()
        neighbor_counts = row_splits[1:] - row_splits[:-1]
        if source_index.numel() == 0:
            return hidden
        target_index = torch.repeat_interleave(torch.arange(search_positions.shape[0], device=search_positions.device), neighbor_counts)
        relative_position = edge_positions[source_index] - edge_positions[target_index]
        distance = torch.linalg.norm(relative_position, dim=-1, keepdim=True)
        sigma_source = sigma_values[source_index].abs().clamp_min(1.0e-8)
        sigma_target = sigma_values[target_index].abs().clamp_min(1.0e-8)
        edge_features = torch.cat([
            hidden[source_index],
            hidden[target_index],
            relative_position,
            distance,
            distance / sigma_source,
            distance / sigma_target,
            gamma_values[source_index],
            gamma_values[target_index],
            sigma_source,
            sigma_target,
        ], dim=-1)
        messages = self.message_network(edge_features)
        aggregated = torch.zeros_like(hidden)
        aggregated.index_add_(0, target_index, messages)
        aggregated = aggregated / neighbor_counts.to(hidden.dtype).clamp_min(1.0).unsqueeze(-1)
        return self.normalization(hidden + aggregated)


class ParticleEdgeGNO(nn.Module):
    """Particle U/gradU model used for stable Task-1 training.

    It predicts all input particles, then gathers the requested query subset. This
    matches the Task-1 dataset where output queries are sampled from input particles.
    """

    uses_output_indices = True

    def __init__(self, in_channels: int, out_channels: int, cfg: Dict, feature_names: Sequence[str], input_mean_np: np.ndarray, input_std_np: np.ndarray):
        super().__init__()
        hidden = int(cfg["edge_hidden_size"])
        dropout = float(cfg["edge_dropout"])
        self.feature_names = list(feature_names)
        self.use_physical_features = bool(cfg["edge_use_physical_features"])
        self.register_buffer("input_mean", torch.as_tensor(input_mean_np, dtype=torch.float32).reshape(1, -1))
        self.register_buffer("input_std", torch.as_tensor(input_std_np, dtype=torch.float32).reshape(1, -1))
        self.sigma_index = self.feature_names.index("sigma") if "sigma" in self.feature_names else -1
        self.gamma_indices = [self.feature_names.index(name) for name in ("Gamma_x", "Gamma_y", "Gamma_z") if name in self.feature_names]
        if len(self.gamma_indices) != 3:
            self.gamma_indices = []
        self.input_network = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.graph_blocks = nn.ModuleList([
            EdgeFeatureMessageBlock(
                hidden_size=hidden,
                neighbor_radius=float(cfg["edge_neighbor_radius"]),
                use_open3d_neighbor_search=bool(cfg["gno_use_open3d"]),
                dropout=dropout,
            )
            for _ in range(int(cfg["edge_graph_layers"]))
        ])
        self.output_network = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )
        self.backend_information = {
            "model_family": "edge_gno",
            "neighbor_radius": float(cfg["edge_neighbor_radius"]),
            "edge_features": "dx, distance, distance/sigma, Gamma source/target, sigma source/target",
        }

    def physical_feature_values(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_physical_features:
            return x
        return x * self.input_std.to(dtype=x.dtype, device=x.device) + self.input_mean.to(dtype=x.dtype, device=x.device)

    def forward(self, input_geom: torch.Tensor, latent_queries: torch.Tensor, output_queries: torch.Tensor, x: torch.Tensor, output_indices: Optional[torch.Tensor] = None, input_geom_physical: Optional[torch.Tensor] = None) -> torch.Tensor:
        outs = []
        for b in range(int(x.shape[0])):
            xb = x[b]
            search_positions = input_geom[b]
            edge_positions = input_geom_physical[b] if input_geom_physical is not None else search_positions
            physical_features = self.physical_feature_values(xb)
            if self.sigma_index >= 0:
                sigma_values = physical_features[:, self.sigma_index:self.sigma_index + 1]
            else:
                sigma_values = torch.ones((xb.shape[0], 1), device=xb.device, dtype=xb.dtype)
            if self.gamma_indices:
                gamma_values = physical_features[:, self.gamma_indices]
            else:
                gamma_values = torch.zeros((xb.shape[0], 3), device=xb.device, dtype=xb.dtype)
            hidden = self.input_network(xb)
            for block in self.graph_blocks:
                hidden = block(hidden, search_positions, edge_positions, gamma_values, sigma_values)
            pred_all = self.output_network(hidden)
            if output_indices is not None:
                pred_all = pred_all[output_indices[b].long()]
            outs.append(pred_all)
        return torch.stack(outs, dim=0)

# %% cell 26
def build_model(cfg: Dict) -> nn.Module:
    if TASK == "particle_ugradu" and cfg["model_family"] == "edge_gno":
        return ParticleEdgeGNO(len(feature_names), len(target_names), cfg, feature_names, input_mean, input_std).to(DEVICE)
    if TASK == "field_reconstruction" and FIELD_DECODER == "pointwise":
        return PointwiseLatentGINO(len(feature_names), len(target_names), cfg).to(DEVICE)
    return build_neuralop_gino(cfg)


MODEL = build_model(CFG)
TOTAL_PARAMS = count_model_params(MODEL)
TRAINABLE_PARAMS = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
print("Latent queries:", tuple(LATENT_QUERIES.shape))
print("Parameters    :", f"{TOTAL_PARAMS:,}", "total /", f"{TRAINABLE_PARAMS:,}", "trainable")
if hasattr(MODEL, "backend_information"):
    print("Backend       :", MODEL.backend_information)

target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
target_std_t = torch.tensor(target_std, dtype=torch.float32, device=DEVICE).view(1, 1, -1)
target_var_t = torch.tensor(target_std**2, dtype=torch.float32, device=DEVICE).view(1, 1, -1).clamp_min(1e-12)

# %% cell 27
def move_batch(batch: Dict) -> Dict:
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict(model: nn.Module, batch: Dict) -> torch.Tensor:
    kwargs = dict(
        input_geom=batch["input_geom"],
        latent_queries=LATENT_QUERIES,
        output_queries=batch["output_queries"],
        x=batch["x"],
    )
    if getattr(model, "uses_output_indices", False) and "output_indices" in batch:
        kwargs["output_indices"] = batch["output_indices"]
    if "input_geom_physical" in batch:
        kwargs["input_geom_physical"] = batch["input_geom_physical"]
    return model(**kwargs)


def denormalize_target(y_norm: torch.Tensor) -> torch.Tensor:
    return y_norm * target_std_t + target_mean_t


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12, minimum_norm: Optional[float] = None) -> torch.Tensor:
    diff = (pred - target).reshape(pred.shape[0], -1)
    ref = target.reshape(target.shape[0], -1)
    ref_norm = torch.linalg.norm(ref, dim=1)
    rel = torch.linalg.norm(diff, dim=1) / ref_norm.clamp_min(eps)
    if minimum_norm is not None:
        rel = torch.where(ref_norm > float(minimum_norm), rel, torch.full_like(rel, torch.nan))
    return rel


def training_task_standard_deviations(frame_ids: np.ndarray) -> Tuple[float, float]:
    velocity_count = gradient_count = 0
    velocity_sum = velocity_square_sum = 0.0
    gradient_sum = gradient_square_sum = 0.0
    for frame_id in np.asarray(frame_ids, dtype=np.int64):
        target = np.asarray(dataset_file["targets_by_frame"][int(frame_id)], dtype=np.float64)
        velocity = target[:, :3].reshape(-1)
        gradient = target[:, 3:].reshape(-1)
        velocity_count += velocity.size
        velocity_sum += float(velocity.sum())
        velocity_square_sum += float(np.dot(velocity, velocity))
        gradient_count += gradient.size
        gradient_sum += float(gradient.sum())
        gradient_square_sum += float(np.dot(gradient, gradient))
    velocity_mean = velocity_sum / max(velocity_count, 1)
    gradient_mean = gradient_sum / max(gradient_count, 1)
    velocity_variance = max(velocity_square_sum / max(velocity_count, 1) - velocity_mean ** 2, 1e-12)
    gradient_variance = max(gradient_square_sum / max(gradient_count, 1) - gradient_mean ** 2, 1e-12)
    return float(np.sqrt(velocity_variance)), float(np.sqrt(gradient_variance))


velocity_task_std_value, gradient_task_std_value = training_task_standard_deviations(train_frame_ids)
velocity_task_std = torch.tensor(velocity_task_std_value, dtype=torch.float32, device=DEVICE)
gradient_task_std = torch.tensor(gradient_task_std_value, dtype=torch.float32, device=DEVICE)
minimum_target_norm_for_relative_error = float(CFG["minimum_target_norm_for_relative_error"])
print("Training velocity std:", f"{velocity_task_std_value:.6e}")
print("Training gradU std   :", f"{gradient_task_std_value:.6e}")


def channel_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if CFG["loss_name"] == "mse":
        return (prediction - target) ** 2
    if CFG["loss_name"] == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, reduction="none", beta=float(CFG["smooth_l1_beta"]))
    if CFG["loss_name"] == "relative_l2":
        numerator = (prediction - target) ** 2
        denominator = torch.mean(target ** 2, dim=(0, 1), keepdim=True).clamp_min(1e-8)
        return numerator / denominator
    raise ValueError(f"Unknown loss_name: {CFG['loss_name']}")


def grouped_training_loss(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if CFG["loss_balance_mode"] == "variance_normalized_physical_mse":
        pred_phys = denormalize_target(pred)
        target_phys = denormalize_target(target)
        velocity_loss = torch.mean((pred_phys[..., :3] - target_phys[..., :3]) ** 2) / velocity_task_std.clamp_min(1e-12).pow(2)
        gradient_loss = torch.mean((pred_phys[..., 3:] - target_phys[..., 3:]) ** 2) / gradient_task_std.clamp_min(1e-12).pow(2)
        return velocity_loss + gradient_loss, velocity_loss.detach(), gradient_loss.detach()

    point_loss = channel_loss(pred, target)
    velocity_loss = point_loss[..., :3].mean()
    gradient_loss = point_loss[..., 3:].mean()
    if CFG["loss_balance_mode"] == "equal_task_mean":
        total = 0.5 * velocity_loss + 0.5 * gradient_loss
    elif CFG["loss_balance_mode"] == "manual":
        total = velocity_loss + gradient_loss
    else:
        raise ValueError(f"Unknown loss_balance_mode: {CFG['loss_balance_mode']}")
    return total, velocity_loss.detach(), gradient_loss.detach()


def particle_cap_for_epoch(epoch_number: int) -> int:
    if int(CFG["epochs"]) <= 1:
        return int(CFG["particles_per_frame_final"])
    progress = (int(epoch_number) - 1) / max(int(CFG["epochs"]) - 1, 1)
    start = int(CFG["particles_per_frame_start"])
    final = int(CFG["particles_per_frame_final"])
    return int(round((1.0 - progress) * start + progress * final))

# %% cell 28
def finite_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else np.nan


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, collect: bool = False, maximum_frames: Optional[int] = None) -> Dict:
    model.eval()
    empty = {"loss": math.nan, "rel_l2_norm": math.nan, "rel_l2_phys": math.nan, "rmse_phys": math.nan, "mae_phys": math.nan, "frames_evaluated": 0, "frames_skipped_for_relative_error": 0, "rel_per_sample": np.asarray([], dtype=np.float32)}
    if len(loader.dataset) == 0:
        return empty
    indices = np.arange(len(loader.dataset), dtype=np.int64)
    if maximum_frames is not None and len(indices) > int(maximum_frames):
        rng = np.random.default_rng(CFG["seed"] + 6701)
        indices = np.sort(rng.choice(indices, size=int(maximum_frames), replace=False))
    losses, rel_norms, rel_phys_values, rmses, maes = [], [], [], [], []
    rels = []
    skipped = 0
    for dataset_index in indices:
        batch = move_batch(loader.dataset[int(dataset_index)])
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) and k != "frame_id" else v) for k, v in batch.items()}
        if torch.is_tensor(batch.get("frame_id")) and batch["frame_id"].ndim == 0:
            batch["frame_id"] = batch["frame_id"].view(1)
        pred = predict(model, batch)
        y = batch["y"]
        loss, _, _ = grouped_training_loss(pred, y)
        pred_phys = denormalize_target(pred)
        y_phys = denormalize_target(y)
        rel_phys = relative_l2(pred_phys, y_phys, minimum_norm=minimum_target_norm_for_relative_error)
        rel_norm = relative_l2(pred, y, minimum_norm=minimum_target_norm_for_relative_error)
        losses.append(float(loss.item()))
        rel_norms.append(float(torch.nanmean(rel_norm).item()) if torch.isfinite(rel_norm).any() else np.nan)
        rel_phys_values.append(float(torch.nanmean(rel_phys).item()) if torch.isfinite(rel_phys).any() else np.nan)
        rmses.append(float(torch.sqrt(torch.mean((pred_phys - y_phys) ** 2)).item()))
        maes.append(float(torch.mean(torch.abs(pred_phys - y_phys)).item()))
        finite_rel = rel_phys[torch.isfinite(rel_phys)]
        if finite_rel.numel() == 0:
            skipped += 1
        elif collect:
            rels.extend(finite_rel.detach().cpu().numpy().tolist())
    out = {
        "loss": finite_mean(losses),
        "rel_l2_norm": finite_mean(rel_norms),
        "rel_l2_phys": finite_mean(rel_phys_values),
        "rmse_phys": finite_mean(rmses),
        "mae_phys": finite_mean(maes),
        "frames_evaluated": int(len(indices)),
        "frames_skipped_for_relative_error": int(skipped),
    }
    if collect:
        out["rel_per_sample"] = np.asarray(rels, dtype=np.float32)
    return out

# %% cell 29
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

# %% cell 30
optimizer = torch.optim.AdamW(MODEL.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(int(CFG["epochs"]), 1),
    eta_min=float(CFG["min_lr"]),
)
use_amp = bool(DEVICE.type == "cuda" and CFG["use_amp"])
mixed_precision_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and mixed_precision_dtype == torch.float16))
except TypeError:  # older torch
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and mixed_precision_dtype == torch.float16))
autocast_options = dict(device_type=DEVICE.type, dtype=mixed_precision_dtype, enabled=use_amp)
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
print("AMP:", use_amp, mixed_precision_dtype)

# %% cell 31
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
        "velocity_task_std": float(velocity_task_std_value),
        "gradient_task_std": float(gradient_task_std_value),
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

# %% cell 32
for epoch in range(1, int(CFG["epochs"]) + 1):
    MODEL.train()
    start_time = time.time()
    particle_cap = particle_cap_for_epoch(epoch)
    if TASK == "particle_ugradu":
        train_ds.max_input_particles = particle_cap
        train_ds.max_output_points = particle_cap
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
            rel = relative_l2(pred, batch["y"], minimum_norm=minimum_target_norm_for_relative_error)
            train_losses.append(float(loss.item()))
            train_rels.append(float(torch.nanmean(rel).item()) if torch.isfinite(rel).any() else np.nan)
            primary_losses.append(float(primary_loss.item()))
            secondary_losses.append(float(secondary_loss.item()))
            good_batches += 1

    scheduler.step()
    if good_batches == 0:
        raise RuntimeError("All training batches were non-finite.")

    if epoch % int(CFG["eval_every"]) == 0 or epoch == int(CFG["epochs"]):
        eval_cap = int(CFG["particles_per_frame_final"])
        if TASK == "particle_ugradu":
            for ds in (val_id_ds, val_angle_ds, test_ds):
                ds.max_input_particles = eval_cap
                ds.max_output_points = eval_cap
        val_id_metrics = evaluate(MODEL, val_id_loader, maximum_frames=CFG["eval_max_frames"])
        val_angle_metrics = evaluate(MODEL, val_angle_loader, maximum_frames=CFG["eval_max_frames"])
        test_metrics = evaluate(MODEL, test_loader, maximum_frames=CFG["eval_max_frames"])
        train_loss = float(np.mean(train_losses))
        train_rel = finite_mean(train_rels)
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
        history.setdefault("particle_cap", []).append(int(particle_cap))
        history.setdefault("val_id_skipped_tiny_target", []).append(int(val_id_metrics["frames_skipped_for_relative_error"]))
        history.setdefault("val_angle_skipped_tiny_target", []).append(int(val_angle_metrics["frames_skipped_for_relative_error"]))
        history.setdefault("test_skipped_tiny_target", []).append(int(test_metrics["frames_skipped_for_relative_error"]))
        metrics_bundle = {
            "train_loss": train_loss,
            "train_rel_l2_norm": train_rel,
            "val_id_rel_l2_phys": val_id_metrics["rel_l2_phys"],
            "val_angle_rel_l2_phys": val_angle_metrics["rel_l2_phys"],
            "test_rel_l2_phys": test_metrics["rel_l2_phys"],
            "test_rmse_phys": test_metrics["rmse_phys"],
            "particle_cap": int(particle_cap),
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
            f"particles={particle_cap} eval_frames={CFG['eval_max_frames']} "
            f"skip_rel={val_id_metrics['frames_skipped_for_relative_error']}/{val_angle_metrics['frames_skipped_for_relative_error']}/{test_metrics['frames_skipped_for_relative_error']} "
            f"good={good_batches} bad={bad_batches} steps={optimizer_steps} "
            f"lr={history['lr'][-1]:.2e} time={elapsed:.1f}s"
        )

# %% cell 33
if not CKPT_PATH.exists():
    fallback_score = history["test_rel_l2_phys"][-1] if history["test_rel_l2_phys"] else float("inf")
    save_checkpoint(CKPT_PATH, "best_fallback", fallback_score)

print("Best score     :", best_score)
print("Best checkpoint:", CKPT_PATH)
print("Last checkpoint:", LAST_CKPT_PATH)
print("History        :", HISTORY_PATH)
