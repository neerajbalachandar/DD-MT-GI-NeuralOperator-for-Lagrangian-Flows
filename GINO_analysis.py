#!/usr/bin/env python3
# =====================================================================
# GINO_analysis.py
#
# Evaluation script for the checkpoint produced by
# GINO_sharedlatent.ipynb.
#
# The field-slice evaluation intentionally follows the previously
# validated notebook protocol:
#   1. Load the SAME checkpoint produced by the training notebook.
#   2. Use the SAME particle-selection rule as training.
#   3. Select the SAME AoA/phase test pairs deterministically.
#   4. Use the SAME physical y-plane definition:
#          y = y_mid + NORM_Y_PLANE * (span_y / 2)
#      with NORM_Y_PLANE = 0.5.
#   5. Load the TRUE velocity field from the native HDF5 field output.
#   6. Query GINO on a dense x-z grid at exactly the SAME y-plane.
#   7. Compare true and predicted |u|/U_inf.
#
# The publication field figure uses only the native HDF5 field. If the
# native field cannot be found, the script stops instead of substituting
# sampled query data.
# =====================================================================

import inspect
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from scipy.interpolate import griddata

from neuralop.layers.gno_block import GNOBlock
from neuralop.models import FNO


# =====================================================================
# 0. USER CONTROLS
# =====================================================================

RUN_TAG = "gino_evolution_sharedlatent_4"

# This is the normalized spanwise definition used by the previously
# validated field-slice plot.
NORM_Y_PLANE = 0.5

# The field figure uses the same three target phases.
SELECTED_PHASES = (0.3, 0.5, 0.7)

# One profile station per selected phase.
PROFILE_X_VALUES = (0.20, 0.40, 0.60)

# Resolution of the predicted field visualization only.
PRED_RESOLUTION = 128

# z clipping for the profile panels.
Z_CLIP_AOA_21 = 2.0
Z_CLIP_AOA_32 = 2.5

# Maximum number of input particles used by training/checkpoint.
MAX_INPUT_PARTICLES = 4096

# Maximum query points used for metric evaluation.
# This does NOT affect the dense field figure, which uses
# PRED_RESOLUTION**2 points.
MAX_METRIC_QUERY_POINTS = 4096

# Set this environment variable when the native HDF5 field directory
# differs from the path used in the original validated notebook.
DEFAULT_FIELD_ROOT = "/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task2"
# Original VPM particle-input root (documented for traceability; the processed NPZ supplies the model inputs used here).
DEFAULT_PARTICLE_INPUT_ROOT = "/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task1"

# Optional explicit overrides:
#   GINO_CHECKPOINT=/path/to/gino_evolution_sharedlatent_4_best_model.pt
#   GINO_DATASET=/path/to/particle_evolution_dataset.npz
#   GINO_FIELD_ROOT=/path/to/task2
#
# =====================================================================


# =====================================================================
# 1. REPRODUCIBILITY / DEVICE / PATH RESOLUTION
# =====================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_repo_root() -> Path:
    """Find the project root without assuming the current working directory."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        dataset_candidates = (
            candidate / "Processed_Data" / "processed_data_split3",
            candidate / "processed_data_split3",
        )
        if any(p.is_dir() for p in dataset_candidates):
            return candidate
    return cwd


REPO_ROOT = find_repo_root()
PLOT_DIR = REPO_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path() -> Path:
    env = os.environ.get("GINO_DATASET", "").strip()
    if env:
        path = Path(env).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"GINO_DATASET does not exist: {path}")
        return path

    candidates = [
        REPO_ROOT / "Processed_Data" / "processed_data_split3" / "particle_evolution_dataset.npz",
        REPO_ROOT / "processed_data_split3" / "particle_evolution_dataset.npz",
    ]
    existing = [p for p in candidates if p.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        # Prefer the path convention used by the current analysis script.
        return existing[0]

    raise FileNotFoundError(
        "Could not find particle_evolution_dataset.npz. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def resolve_checkpoint_path() -> Path:
    """
    Prefer the exact path used by GINO_sharedlatent.ipynb.

    The training notebook uses:
        pt_model/sharedlatent_GINO_4/
        gino_evolution_sharedlatent_4_best_model.pt

    An uppercase Pt_Model fallback is allowed only when the exact training
    path is absent. If multiple additional matches exist, fail rather than
    silently load an arbitrary checkpoint.
    """
    env = os.environ.get("GINO_CHECKPOINT", "").strip()
    if env:
        path = Path(env).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"GINO_CHECKPOINT does not exist: {path}")
        return path

    exact_training = (
        REPO_ROOT
        / "pt_model"
        / "sharedlatent_GINO_4"
        / f"{RUN_TAG}_best_model.pt"
    )
    uppercase_fallback = (
        REPO_ROOT
        / "Pt_Model"
        / "sharedlatent_GINO_4"
        / f"{RUN_TAG}_best_model.pt"
    )

    if exact_training.exists():
        return exact_training
    if uppercase_fallback.exists():
        print(
            "WARNING: training-path checkpoint was not found; using the "
            "uppercase Pt_Model fallback:"
        )
        print(f"         {uppercase_fallback}")
        return uppercase_fallback

    matches = list(REPO_ROOT.rglob(f"{RUN_TAG}_best_model.pt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple matching checkpoints were found. Set GINO_CHECKPOINT "
            "explicitly rather than loading an arbitrary file:\n"
            + "\n".join(str(p) for p in matches)
        )

    raise FileNotFoundError(
        "Could not find the GINO checkpoint. Expected:\n"
        f"  {exact_training}\n"
        f"or:\n  {uppercase_fallback}\n"
        "Set GINO_CHECKPOINT to an explicit path if necessary."
    )


DATASET_PATH = resolve_dataset_path()
CKPT_PATH = resolve_checkpoint_path()

FIELD_ROOT = Path(
    os.environ.get("GINO_FIELD_ROOT", DEFAULT_FIELD_ROOT)
).expanduser()

print("=" * 72)
print(f"Device       : {DEVICE}")
print(f"Repository   : {REPO_ROOT}")
print(f"Checkpoint   : {CKPT_PATH}")
print(f"Dataset      : {DATASET_PATH}")
print(f"Native fields: {FIELD_ROOT}")
print("=" * 72)


# =====================================================================
# 2. LOAD CHECKPOINT + DATASET
# =====================================================================

checkpoint = torch.load(
    CKPT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

cfg = dict(checkpoint["config"])

if "checkpoint_tag" in checkpoint and checkpoint["checkpoint_tag"] != RUN_TAG:
    raise RuntimeError(
        f"Checkpoint tag mismatch: expected {RUN_TAG!r}, "
        f"got {checkpoint['checkpoint_tag']!r}."
    )

data = np.load(DATASET_PATH, allow_pickle=True)

feature_names_all = [
    str(x) for x in np.asarray(data["feature_names"]).tolist()
]

feature_names = [
    str(x) for x in checkpoint["feature_names"]
]

target_names = [
    str(x) for x in checkpoint["target_names"]
]

field_target_names = [
    str(x) for x in checkpoint["field_target_names"]
]

if len(field_target_names) < 3 or field_target_names[:3] != ["u_x", "u_y", "u_z"]:
    raise RuntimeError(
        "Checkpoint field targets must begin with u_x, u_y, u_z; "
        f"got {field_target_names}"
    )

required_features = [
    "x", "y", "z",
    "u_x", "u_y", "u_z",
    "Gamma_x", "Gamma_y", "Gamma_z",
    "sigma",
    "gradU_xx", "gradU_xy", "gradU_xz",
    "gradU_yx", "gradU_yy", "gradU_yz",
    "gradU_zx", "gradU_zy", "gradU_zz",
]

missing = [name for name in required_features if name not in feature_names_all]
if missing:
    raise KeyError(
        "The dataset is missing required feature channels: "
        + ", ".join(missing)
    )

active_input_indices = [
    feature_names_all.index(name) for name in feature_names
]

coord_indices = [
    feature_names_all.index(name) for name in ("x", "y", "z")
]

velocity_indices = [
    feature_names_all.index(name) for name in ("u_x", "u_y", "u_z")
]

gradient_indices = [
    feature_names_all.index(f"gradU_{i}{j}")
    for i in "xyz"
    for j in "xyz"
]

state_indices = [
    feature_names_all.index(name)
    for name in ("x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma")
]

# Exact aliases used by the training notebook's rollout reconstruction.
# They point to the same integer indices already computed above.
active_input_feature_indices = active_input_indices
state_feature_indices = state_indices
velocity_feature_indices = velocity_indices
gradient_feature_indices = gradient_indices

inputs_t = np.asarray(data["inputs_t"], dtype=np.float32)

targets_delta_norm = np.asarray(
    data[
        "targets_residual_norm"
        if "targets_residual_norm" in data.files
        else "targets_delta_norm"
    ],
    dtype=np.float32,
)

targets_delta_raw = np.asarray(
    data["targets_delta"],
    dtype=np.float32,
)

query_coords_all = np.asarray(
    data["query_coords"],
    dtype=np.float32,
)

targets_field_norm = np.asarray(
    data["targets_velocity_field_norm"],
    dtype=np.float32,
)

field_mask = np.asarray(
    data["field_query_mask"],
    dtype=bool,
)

frame_ranges = list(
    data["pair_ranges"]
    if "pair_ranges" in data.files
    else data["frame_ranges"]
)

frame_contexts = list(
    data["pair_contexts"]
    if "pair_contexts" in data.files
    else data["frame_contexts"]
)

train_ids = np.asarray(
    data["train_pair_ids"] if "train_pair_ids" in data.files else [],
    dtype=np.int64,
)

val_ids = np.asarray(
    data["val_pair_ids"] if "val_pair_ids" in data.files else [],
    dtype=np.int64,
)

test_ids = np.asarray(
    data["test_pair_ids"] if "test_pair_ids" in data.files else [],
    dtype=np.int64,
)

# The checkpoint stores the training normalization actually used.
input_mean = np.asarray(checkpoint["input_mean"], dtype=np.float32).reshape(-1)
input_std = np.maximum(
    np.asarray(checkpoint["input_std"], dtype=np.float32).reshape(-1),
    1e-8,
)

target_mean = np.asarray(
    checkpoint["target_mean"], dtype=np.float32
).reshape(-1)

target_std = np.maximum(
    np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(-1),
    1e-8,
)

field_mean = np.asarray(
    checkpoint["field_mean"], dtype=np.float32
).reshape(-1)

field_std = np.maximum(
    np.asarray(checkpoint["field_std"], dtype=np.float32).reshape(-1),
    1e-8,
)

coord_min = np.asarray(
    checkpoint["coord_min"], dtype=np.float32
).reshape(3)

coord_span = np.maximum(
    np.asarray(checkpoint["coord_span"], dtype=np.float32).reshape(3),
    1e-8,
)

# Fixed physical field-slice location. The validated field-plot protocol
# defines the plane relative to the spanwise coordinate range as
# y_plane = y_mid + NORM_Y_PLANE * (span_y / 2), with NORM_Y_PLANE=0.5.
y_mid = float(coord_min[1] + 0.5 * coord_span[1])
y_half_span = float(0.5 * coord_span[1])
Y_PLANE = float(y_mid + NORM_Y_PLANE * y_half_span)

print(
    f"Validated field slice: y = {Y_PLANE:.8f} "
    f"(NORM_Y_PLANE = {NORM_Y_PLANE})"
)

global_condition_channels = list(
    checkpoint.get(
        "global_condition_channels",
        cfg["global_condition_channels"],
    )
)

delta_channels = len(target_names)
field_channels = len(field_target_names)


if len(input_mean) != len(feature_names):
    raise RuntimeError(
        f"Input normalization has length {len(input_mean)}, "
        f"but checkpoint uses {len(feature_names)} input features."
    )

if len(input_std) != len(feature_names):
    raise RuntimeError(
        f"Input std has length {len(input_std)}, "
        f"but checkpoint uses {len(feature_names)} input features."
    )

if len(target_mean) < delta_channels or len(target_std) < delta_channels:
    raise RuntimeError(
        "Target normalization arrays are shorter than the checkpoint's "
        "delta output dimension."
    )

if len(field_mean) != field_channels or len(field_std) != field_channels:
    raise RuntimeError(
        "Field normalization dimension does not match field output channels."
    )


input_mean_t = torch.tensor(input_mean, dtype=torch.float32, device=DEVICE)
input_std_t = torch.tensor(input_std, dtype=torch.float32, device=DEVICE)
target_mean_t = torch.tensor(
    target_mean[:delta_channels],
    dtype=torch.float32,
    device=DEVICE,
)
target_std_t = torch.tensor(
    target_std[:delta_channels],
    dtype=torch.float32,
    device=DEVICE,
)
field_mean_t = torch.tensor(field_mean, dtype=torch.float32, device=DEVICE)
field_std_t = torch.tensor(field_std, dtype=torch.float32, device=DEVICE)


def as_context(obj) -> Dict:
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "item"):
        item = obj.item()
        if isinstance(item, dict):
            return item

    return dict(obj)


pair_context_map = {
    int(i): as_context(frame_contexts[int(i)])
    for i in range(len(frame_contexts))
}


# Reproduce the training particle-selection rule exactly:
# for each case, take the minimum particle count over all case frames,
# then use the first n_use particles.
case_to_nmin = {}

for pair_id in range(len(frame_ranges)):
    case = str(pair_context_map[pair_id].get("case", ""))
    n = int(frame_ranges[pair_id][5])

    if case not in case_to_nmin:
        case_to_nmin[case] = n
    else:
        case_to_nmin[case] = min(case_to_nmin[case], n)


print(f"Input features : {feature_names}")
print(f"Delta targets  : {target_names}")
print(f"Field targets  : {field_target_names}")
print(f"Global channels: {global_condition_channels}")
print(f"Test pairs     : {len(test_ids)}")
print(f"Case n_min     : {case_to_nmin}")


# =====================================================================
# 3. OPTIONAL ROLLOUT METADATA
#
# These are only needed if a checkpoint was trained with lookback_steps > 0.
# For the current shared-latent checkpoint this should normally be zero.
# =====================================================================

rollout_cases = [
    str(x)
    for x in np.asarray(
        data["rollout_cases"] if "rollout_cases" in data.files else []
    ).tolist()
]

rollout_true_states = list(
    data["rollout_true_states"]
    if "rollout_true_states" in data.files
    else []
)

rollout_phases = list(
    data["rollout_phases"]
    if "rollout_phases" in data.files
    else []
)

case_to_rollout_index = {
    case: i for i, case in enumerate(rollout_cases)
}

case_to_frame_ids_from_pairs = defaultdict(list)
case_frame_to_row = {}

for pair_id, ctx in pair_context_map.items():
    case = str(ctx.get("case", ""))
    frame_id = str(
        ctx.get("frame_t", ctx.get("frame_id", ""))
    ).zfill(6)

    if not case:
        continue

    row = frame_ranges[pair_id]
    case_frame_to_row[(case, frame_id)] = (
        int(row[3]),
        int(row[4]),
    )
    case_to_frame_ids_from_pairs[case].append(frame_id)

case_to_frame_index = {}
case_rollout_index_to_frame = {}

if "rollout_frame_ids" in data.files:
    for case_idx, case in enumerate(rollout_cases):
        frame_ids = [
            str(x).zfill(6)
            for x in data["rollout_frame_ids"][case_idx].tolist()
        ]
        case_to_frame_index[case] = {
            fid: j for j, fid in enumerate(frame_ids)
        }
        case_rollout_index_to_frame[case] = {
            j: fid for j, fid in enumerate(frame_ids)
        }
else:
    for case_idx, case in enumerate(rollout_cases):
        seq = np.asarray(rollout_true_states[case_idx])
        frame_ids = sorted(
            set(case_to_frame_ids_from_pairs.get(case, []))
        )

        if len(frame_ids) >= seq.shape[0]:
            frame_ids = frame_ids[:seq.shape[0]]
        else:
            frame_ids = [
                str(j).zfill(6)
                for j in range(seq.shape[0])
            ]

        case_to_frame_index[case] = {
            fid: j for j, fid in enumerate(frame_ids)
        }
        case_rollout_index_to_frame[case] = {
            j: fid for j, fid in enumerate(frame_ids)
        }


# =====================================================================
# 4. NORMALIZATION / CONTEXT HELPERS
# =====================================================================

def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    return np.clip(
        (xyz - coord_min[None, :]) / coord_span[None, :],
        0.0,
        1.0,
    ).astype(np.float32)


def normalize_geom_t(xyz: torch.Tensor) -> torch.Tensor:
    cmin = torch.tensor(
        coord_min,
        dtype=torch.float32,
        device=xyz.device,
    ).view(1, 1, 3)

    cspan = torch.tensor(
        coord_span,
        dtype=torch.float32,
        device=xyz.device,
    ).view(1, 1, 3)

    return torch.clamp((xyz - cmin) / cspan, 0.0, 1.0)


def context_global_params(
    context: Dict,
    phase: Optional[float] = None,
) -> np.ndarray:
    freestream = np.asarray(
        context.get("freestream", [0.0, 0.0, 0.0]),
        dtype=np.float32,
    ).reshape(-1)

    if freestream.size < 3:
        freestream = np.pad(
            freestream,
            (0, 3 - freestream.size),
        )

    values = {
        "angle_of_attack": float(
            context.get("aoa_deg", 0.0)
        ) / 45.0,
        "freestream_magnitude": float(
            np.linalg.norm(freestream)
        ) / 10.0,
        "freestream_x": float(freestream[0]) / 10.0,
        "freestream_y": float(freestream[1]) / 10.0,
        "freestream_z": float(freestream[2]) / 10.0,
        "phase": float(
            context.get("phase_t", 0.0)
            if phase is None
            else phase
        ),
    }

    return np.asarray(
        [values[name] for name in global_condition_channels],
        dtype=np.float32,
    )


# =====================================================================
# 5. EXACT TRAINING-MODEL ARCHITECTURE
# =====================================================================

def make_gnoblock(
    in_channels: int,
    out_channels: int,
    radius: float,
):
    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        coord_dim=3,
        radius=float(radius),
        transform_type="linear",
        reduction="mean",
        pos_embedding_type="transformer",
        pos_embedding_channels=12,
        channel_mlp_layers=[
            out_channels,
            out_channels,
            out_channels,
        ],
    )

    accepted = set(
        inspect.signature(GNOBlock.__init__).parameters
    )

    if "use_torch_scatter_reduce" in accepted:
        kwargs["use_torch_scatter_reduce"] = False

    if "use_open3d_neighbor_search" in accepted:
        kwargs["use_open3d_neighbor_search"] = False

    return GNOBlock(
        **{
            k: v
            for k, v in kwargs.items()
            if k in accepted
        }
    )


def positional_encoding(
    coords: torch.Tensor,
    num_frequencies: int,
) -> torch.Tensor:
    if int(num_frequencies) <= 0:
        return coords

    freqs = (
        2.0 ** torch.arange(
            int(num_frequencies),
            device=coords.device,
            dtype=coords.dtype,
        )
    ).view(1, 1, -1)

    angles = coords.unsqueeze(-1) * freqs * math.pi

    return torch.cat(
        [
            coords,
            torch.sin(angles).flatten(-2),
            torch.cos(angles).flatten(-2),
        ],
        dim=-1,
    )


def make_latent_queries(
    res: int,
    device: torch.device,
) -> torch.Tensor:
    line = torch.linspace(
        0.0,
        1.0,
        int(res),
        dtype=torch.float32,
        device=device,
    )

    xx, yy, zz = torch.meshgrid(
        line,
        line,
        line,
        indexing="ij",
    )

    return torch.stack(
        [xx, yy, zz],
        dim=-1,
    ).reshape(1, -1, 3)


def get_past_states(batch, lookback=2):
    """
    Exact temporal-input construction used by the training notebook.
    It is inactive when cfg["lookback_steps"] == 0.
    """
    lookback = int(lookback)

    if lookback <= 0 or not rollout_true_states:
        return None

    pair_id = int(
        batch["pair_id"].reshape(-1)[0].item()
    )

    ctx = pair_context_map[pair_id]
    case = str(ctx.get("case", ""))
    frame_t = str(
        ctx.get("frame_t", ctx.get("frame_id", ""))
    ).zfill(6)

    case_idx = case_to_rollout_index.get(case)
    frame_map = case_to_frame_index.get(case, {})
    inv_frame_map = case_rollout_index_to_frame.get(case, {})
    start_idx = frame_map.get(frame_t)

    if (
        case_idx is None
        or start_idx is None
        or start_idx < lookback
    ):
        return None

    seq = np.asarray(
        rollout_true_states[case_idx],
        dtype=np.float32,
    )

    n_batch = int(
        batch["state_phys"].shape[1]
    )

    past = []

    for k in range(lookback, 0, -1):
        ridx = start_idx - k

        frame_id = inv_frame_map.get(
            ridx,
            str(ridx).zfill(6),
        )

        row = case_frame_to_row.get(
            (case, frame_id)
        )

        if row is None:
            return None

        rs, re = row
        rows = inputs_t[rs:re]

        n_use = min(
            n_batch,
            seq.shape[1],
            rows.shape[0],
        )

        state_k = seq[ridx, :n_use, :7]

        vel_k = rows[
            :n_use,
            velocity_indices
        ]

        feat_k = np.concatenate(
            [state_k, vel_k],
            axis=-1,
        ).astype(np.float32)

        if n_use < n_batch:
            return None

        past.append(feat_k)

    return torch.from_numpy(
        np.stack(past, axis=1)
    ).unsqueeze(0).to(DEVICE)


class GINOSharedLatent(nn.Module):
    """
    Architecture reproduced from the training notebook.

    Keeping this class synchronized with the training definition is
    essential because the checkpoint contains the learned weights but
    not the Python class implementation.
    """

    def __init__(
        self,
        in_channels,
        delta_channels,
        field_channels,
        global_channels,
        cfg_local,
    ):
        super().__init__()

        hidden = int(cfg_local["hidden_channels"])

        self.latent_res = int(
            cfg_local["latent_res"]
        )

        self.query_pe_freqs = int(
            cfg_local["query_pos_encoding_frequencies"]
        )

        self.lookback_steps = int(
            cfg_local.get("lookback_steps", 0)
        )

        self.hidden = hidden

        self.lift = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.global_condition_mlp = nn.Sequential(
            nn.Linear(global_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.encoder = make_gnoblock(
            hidden,
            hidden,
            cfg_local["gno_radius"],
        )

        modes = min(
            int(cfg_local["fno_modes"]),
            max(self.latent_res // 2, 1),
        )

        self.fno = FNO(
            n_modes=(modes, modes, modes),
            in_channels=hidden,
            out_channels=hidden,
            hidden_channels=hidden,
            n_layers=int(cfg_local["fno_layers"]),
            positional_embedding=None,
        )

        # ---------------- Field decoder ----------------
        query_dim = (
            3 + 2 * 3 * self.query_pe_freqs
        )

        self.particle_pos_enc_proj = nn.Linear(
            query_dim,
            hidden,
        )

        self.grid_pos_enc_proj = nn.Linear(
            query_dim,
            hidden,
        )

        width = int(
            cfg_local["mlp_hidden"]
        )

        layers = []

        for layer_id in range(
            max(
                int(cfg_local["mlp_layers"]),
                1,
            )
        ):
            layers += [
                nn.Linear(
                    (
                        hidden + query_dim
                        if layer_id == 0
                        else width
                    ),
                    width,
                ),
                nn.GELU(),
            ]

        layers.append(
            nn.Linear(
                width,
                field_channels,
            )
        )

        self.field_decoder = nn.Sequential(*layers)

        # ---------------- Delta decoder ----------------
        self.particle_skip_proj = nn.Linear(
            hidden,
            hidden,
        )

        self.particle_query_proj = nn.Linear(
            hidden,
            hidden,
        )

        self.grid_kv_proj = nn.Linear(
            hidden,
            2 * hidden,
        )

        self.attn_dropout = nn.Dropout(0.0)

        self.attn_norm = nn.LayerNorm(hidden)

        self.delta_fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, delta_channels),
        )

        # ---------------- Optional temporal encoder ----------------
        if self.lookback_steps > 0:
            temporal_in_dim = (
                self.lookback_steps * 10
            )

            self.temporal_encoder_proj = nn.Linear(
                temporal_in_dim,
                hidden,
            )

            self.temporal_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=4,
                    dim_feedforward=256,
                    dropout=0.1,
                    batch_first=True,
                ),
                num_layers=2,
            )

            self.temporal_proj = nn.Linear(
                hidden,
                hidden,
            )
        else:
            self.temporal_encoder = None

        self.log_delta_var = nn.Parameter(
            torch.zeros(())
        )

        self.log_field_var = nn.Parameter(
            torch.zeros(())
        )

    def apply_gno(
        self,
        block,
        source_coords,
        query_coords,
        source_features,
    ):
        return block(
            y=source_coords,
            x=query_coords,
            f_y=source_features,
        )

    def encode_process(
        self,
        input_geom,
        latent_queries,
        x,
        global_params,
    ):
        base_latent = latent_queries[0]
        r = self.latent_res

        grids = []
        flat_latents = []
        particle_skips = []

        for b in range(x.shape[0]):
            h_raw = self.lift(x[b])

            latent = self.apply_gno(
                self.encoder,
                source_coords=input_geom[b],
                query_coords=base_latent,
                source_features=h_raw,
            )

            if latent.ndim == 3:
                latent = latent.squeeze(0)

            grid = latent.reshape(
                r,
                r,
                r,
                -1,
            ).permute(
                3,
                0,
                1,
                2,
            ).unsqueeze(0)

            cond = self.global_condition_mlp(
                global_params[b]
            ).view(
                1,
                -1,
                1,
                1,
                1,
            )

            processed_grid = self.fno(
                grid + cond
            )

            processed_flat = (
                processed_grid
                .squeeze(0)
                .permute(1, 2, 3, 0)
                .reshape(
                    -1,
                    processed_grid.shape[1],
                )
            )

            grids.append(processed_grid)
            flat_latents.append(processed_flat)

            particle_skips.append(
                self.particle_skip_proj(h_raw)
            )

        return (
            base_latent,
            grids,
            flat_latents,
            particle_skips,
        )

    def sample_grid(
        self,
        grid,
        queries,
    ):
        q = queries.clamp(0.0, 1.0)

        sample_grid = (
            (q * 2.0 - 1.0)
            .view(1, -1, 1, 1, 3)
        )

        sampled = F.grid_sample(
            grid,
            sample_grid,
            align_corners=True,
            mode="bilinear",
        )

        return (
            sampled
            .squeeze(0)
            .squeeze(-1)
            .squeeze(-1)
            .transpose(0, 1)
        )

    def cross_attention(
        self,
        queries,
        keys,
        values,
    ):
        d = queries.shape[-1]

        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        ) / math.sqrt(d)

        attn_weights = F.softmax(
            scores,
            dim=-1,
        )

        attn_weights = self.attn_dropout(
            attn_weights
        )

        return torch.matmul(
            attn_weights,
            values,
        )

    def temporal_latent_from_batch(
        self,
        batch_dict,
    ):
        if (
            self.lookback_steps <= 0
            or batch_dict is None
        ):
            return None

        past_feats = get_past_states(
            batch_dict,
            self.lookback_steps,
        )

        if past_feats is None:
            return None

        B, N, L, D = past_feats.shape

        past_flat = past_feats.reshape(
            B,
            N,
            L * D,
        )

        past_embedded = self.temporal_encoder_proj(
            past_flat
        )

        temporal_latent = self.temporal_encoder(
            past_embedded
        )

        return self.temporal_proj(
            temporal_latent
        )

    def delta_from_latents(
        self,
        base_latent,
        flat_latents,
        particle_skips,
        input_geom,
        batch_dict=None,
    ):
        temporal_latent = (
            self.temporal_latent_from_batch(
                batch_dict
            )
        )

        grid_pos_enc = positional_encoding(
            base_latent.unsqueeze(0),
            self.query_pe_freqs,
        ).squeeze(0)

        delta_outputs = []

        for b, (
            flat_latent,
            skip,
        ) in enumerate(
            zip(
                flat_latents,
                particle_skips,
            )
        ):
            pos_enc = positional_encoding(
                input_geom[b].unsqueeze(0),
                self.query_pe_freqs,
            ).squeeze(0)

            q_base = (
                skip
                + self.particle_pos_enc_proj(
                    pos_enc
                )
            )

            if temporal_latent is not None:
                q_base = (
                    q_base
                    + temporal_latent[b]
                )

            q = self.particle_query_proj(
                q_base
            )

            kv_input = (
                flat_latent
                + self.grid_pos_enc_proj(
                    grid_pos_enc
                )
            )

            kv = self.grid_kv_proj(
                kv_input
            )

            k = kv[..., :self.hidden]
            v = kv[..., self.hidden:]

            particle_latent = self.cross_attention(
                q,
                k,
                v,
            )

            particle_latent = (
                self.attn_norm(
                    particle_latent + q
                )
            )

            fused = self.delta_fusion(
                torch.cat(
                    [
                        particle_latent,
                        skip,
                    ],
                    dim=-1,
                )
            )

            delta_outputs.append(
                self.delta_head(
                    fused
                ).unsqueeze(0)
            )

        return torch.cat(
            delta_outputs,
            dim=0,
        )

    def forward(
        self,
        input_geom,
        latent_queries,
        output_queries,
        x,
        global_params,
        batch_dict=None,
    ):
        (
            base_latent,
            grids,
            flat_latents,
            particle_skips,
        ) = self.encode_process(
            input_geom,
            latent_queries,
            x,
            global_params,
        )

        delta = self.delta_from_latents(
            base_latent,
            flat_latents,
            particle_skips,
            input_geom,
            batch_dict=batch_dict,
        )

        field_outputs = []

        for b, grid in enumerate(grids):
            q_field = output_queries[b].clamp(
                0.0,
                1.0,
            )

            sampled = self.sample_grid(
                grid,
                q_field,
            )

            field_out = self.field_decoder(
                torch.cat(
                    [
                        sampled,
                        positional_encoding(
                            q_field.unsqueeze(0),
                            self.query_pe_freqs,
                        ).squeeze(0),
                    ],
                    dim=-1,
                )
            )

            field_outputs.append(
                field_out.unsqueeze(0)
            )

        field = torch.cat(
            field_outputs,
            dim=0,
        )

        return delta, field


LATENT_QUERIES = make_latent_queries(
    cfg["latent_res"],
    DEVICE,
)

model = GINOSharedLatent(
    len(feature_names),
    delta_channels,
    field_channels,
    len(global_condition_channels),
    cfg,
).to(DEVICE)

state_dict = checkpoint["model_state_dict"]

# Some torch checkpoints carry this auxiliary key.
state_dict = dict(state_dict)
state_dict.pop("_metadata", None)

missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False,
)

if missing_keys or unexpected_keys:
    raise RuntimeError(
        "Checkpoint/model mismatch.\n"
        f"Missing keys   : {missing_keys}\n"
        f"Unexpected keys: {unexpected_keys}"
    )

model.eval()

print(
    f"Loaded model with "
    f"{sum(p.numel() for p in model.parameters()):,} parameters."
)

print(
    f"Model lookback_steps = "
    f"{model.lookback_steps}"
)


# =====================================================================
# 6. BATCH CONSTRUCTION
# =====================================================================

def build_batch(
    pair_id: int,
    max_particles: int = MAX_INPUT_PARTICLES,
    max_queries: int = MAX_METRIC_QUERY_POINTS,
):
    """
    Construct a single-pair inference batch using the exact training
    particle selection:

        n_use = min(n_total, n_min_for_case, max_particles)
        idx   = arange(n_use)

    Field metric queries are selected deterministically rather than with
    random sampling so repeated evaluation produces the same result.
    """
    pair_id = int(pair_id)

    start, end = (
        int(frame_ranges[pair_id][3]),
        int(frame_ranges[pair_id][4]),
    )

    raw = inputs_t[start:end]

    ctx = pair_context_map[pair_id]
    case = str(ctx.get("case", "unknown"))

    n_total = raw.shape[0]

    nmin = int(
        case_to_nmin.get(
            case,
            n_total,
        )
    )

    n_use = min(
        n_total,
        nmin,
        int(max_particles),
    )

    idx = np.arange(
        n_use,
        dtype=np.int64,
    )

    x_raw = raw[idx][:, active_input_indices]

    if x_raw.shape != (n_use, len(active_input_indices)):
        raise RuntimeError(
            f"Input feature extraction produced shape {x_raw.shape}; "
            f"expected {(n_use, len(active_input_indices))}."
        )

    x_norm = (
        x_raw - input_mean[None, :]
    ) / input_std[None, :]

    x_norm = np.clip(
        np.nan_to_num(x_norm),
        -8.0,
        8.0,
    ).astype(np.float32)

    coord_raw = raw[idx][:, coord_indices]

    input_geom = normalize_xyz(
        coord_raw
    )

    valid = np.flatnonzero(
        field_mask[pair_id]
    )

    if valid.size == 0:
        raise RuntimeError(
            f"pair_id={pair_id} has no valid field query points."
        )

    if (
        max_queries is None
        or int(max_queries) <= 0
        or valid.size <= int(max_queries)
    ):
        query_idx = valid
    else:
        query_positions = np.linspace(
            0,
            valid.size - 1,
            int(max_queries),
            dtype=np.int64,
        )
        query_idx = valid[query_positions]

    q_xyz = np.asarray(
        query_coords_all[
            pair_id,
            query_idx,
        ],
        dtype=np.float32,
    )

    y_field = np.asarray(
        targets_field_norm[
            pair_id,
            query_idx,
        ],
        dtype=np.float32,
    )

    state_phys = np.asarray(
        raw[idx][:, state_indices],
        dtype=np.float32,
    )

    velocity_phys = np.asarray(
        raw[idx][:, velocity_indices],
        dtype=np.float32,
    )

    gradU_phys = np.asarray(
        raw[idx][:, gradient_indices],
        dtype=np.float32,
    )

    y_delta_norm = np.asarray(
        targets_delta_norm[start:end][idx][:, :delta_channels],
        dtype=np.float32,
    )

    y_delta_raw = np.asarray(
        targets_delta_raw[start:end][idx][:, :delta_channels],
        dtype=np.float32,
    )

    return {
        "input_geom": torch.from_numpy(
            input_geom
        ).unsqueeze(0).to(DEVICE),

        "x": torch.from_numpy(
            x_norm
        ).unsqueeze(0).to(DEVICE),

        "output_queries": torch.from_numpy(
            normalize_xyz(q_xyz)
        ).unsqueeze(0).to(DEVICE),

        "global_params": torch.from_numpy(
            context_global_params(ctx)
        ).unsqueeze(0).to(DEVICE),

        "state_phys": state_phys,
        "velocity_phys": velocity_phys,
        "gradU_phys": gradU_phys,

        "y_delta_norm": y_delta_norm,
        "y_delta_raw": y_delta_raw,
        "y_field_norm": y_field,

        "query_xyz_raw": q_xyz,
        "input_xyz_raw": coord_raw,

        "dt": float(
            ctx.get("dt", 0.0034)
        ),

        "pair_id": torch.tensor(
            [pair_id],
            dtype=torch.long,
            device=DEVICE,
        ),

        "case": case,
    }


def predict(batch):
    return model(
        batch["input_geom"],
        LATENT_QUERIES,
        batch["output_queries"],
        batch["x"],
        batch["global_params"],
        batch_dict=batch,
    )


# =====================================================================
# 7. PHYSICAL DENORMALIZATION / STATE METRICS
# =====================================================================

def denormalize_delta(y):
    return (
        y * target_std_t
        + target_mean_t
    )


def denormalize_field(y):
    return (
        y * field_std_t
        + field_mean_t
    )


def euler_baseline_from_batch(batch):
    state_t = torch.from_numpy(
        batch["state_phys"]
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    u_true = torch.from_numpy(
        batch["velocity_phys"]
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    gradU_true = torch.from_numpy(
        batch["gradU_phys"]
    ).to(
        DEVICE,
        dtype=torch.float32,
    ).reshape(
        -1,
        3,
        3,
    )

    dt = float(
        batch.get(
            "dt",
            0.0034,
        )
    )

    physics_dx = (
        dt * u_true
    )

    physics_dgamma = (
        dt
        * torch.einsum(
            "nij,nj->ni",
            gradU_true,
            state_t[:, 3:6],
        )
    )

    return (
        state_t,
        physics_dx,
        physics_dgamma,
    )


def reconstruct_next_state_from_residual(
    batch,
    residual_pred_norm,
):
    residual_pred_phys = denormalize_delta(
        residual_pred_norm
    )

    state_t, physics_dx, physics_dgamma = (
        euler_baseline_from_batch(batch)
    )

    state_tp1_pred = state_t.clone()

    state_tp1_pred[:, :3] += (
        physics_dx
        + residual_pred_phys[:, :3]
    )

    state_tp1_pred[:, 3:6] += (
        physics_dgamma
        + residual_pred_phys[:, 3:6]
    )

    state_tp1_pred[:, 6] += (
        residual_pred_phys[:, 6]
    )

    raw_delta = torch.from_numpy(
        batch["y_delta_raw"][:, :7]
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    state_tp1_true = (
        state_t + raw_delta
    )

    return (
        state_tp1_pred,
        state_tp1_true,
        residual_pred_phys,
    )


def state_relative_l2(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> float:
    numerator = torch.linalg.norm(
        (pred - true).reshape(
            pred.shape[0],
            -1,
        ),
        dim=1,
    )

    denominator = torch.linalg.norm(
        true.reshape(
            true.shape[0],
            -1,
        ),
        dim=1,
    ).clamp_min(1e-8)

    return float(
        (numerator / denominator)
        .mean()
        .item()
    )


def nrmse(
    pred: np.ndarray,
    true: np.ndarray,
) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)

    return float(
        np.linalg.norm(pred - true)
        / (
            np.linalg.norm(true)
            + 1e-8
        )
    )


# =====================================================================
# 8. GLOBAL TEST-SET EVALUATION
# =====================================================================

@torch.inference_mode()
def evaluate_per_pair():
    state_errors = []
    field_errors_norm = []

    for pid in test_ids:
        pid = int(pid)

        batch = build_batch(
            pid,
            max_particles=MAX_INPUT_PARTICLES,
            max_queries=MAX_METRIC_QUERY_POINTS,
        )

        delta_pred, field_pred = predict(batch)

        delta_pred = delta_pred[0]
        field_pred = field_pred[0]

        state_pred, state_true, _ = (
            reconstruct_next_state_from_residual(
                batch,
                delta_pred,
            )
        )

        state_errors.append(
            state_relative_l2(
                state_pred,
                state_true,
            )
        )

        y_field_norm = torch.from_numpy(
            batch["y_field_norm"]
        ).to(
            DEVICE,
            dtype=torch.float32,
        )

        field_err = (
            torch.linalg.norm(
                (
                    field_pred.reshape(-1)
                    - y_field_norm.reshape(-1)
                )
            )
            /
            torch.linalg.norm(
                y_field_norm.reshape(-1)
            ).clamp_min(1e-8)
        )

        field_errors_norm.append(
            float(field_err.item())
        )

    return (
        np.asarray(state_errors),
        np.asarray(field_errors_norm),
    )


state_errs, field_errs = evaluate_per_pair()

print("=" * 72)
print(f"Test pairs                         : {len(test_ids)}")
print(
    f"State relative L2 mean             : "
    f"{state_errs.mean():.6f}"
)
print(
    f"State relative L2 median           : "
    f"{np.median(state_errs):.6f}"
)
print(
    f"Field normalized relative L2 mean  : "
    f"{field_errs.mean():.6f}"
)
print(
    f"Field normalized relative L2 median: "
    f"{np.median(field_errs):.6f}"
)
print("=" * 72)


# =====================================================================
# =====================================================================
# 9. FIGURE A — ONE-STEP TRANSITION ACCURACY
# =====================================================================
#
# Primary metric:
#
#     RMSE_position =
#       sqrt( mean_i ||x_pred_i - x_true_i||_2^2 )
#
#     RMSE_circulation =
#       sqrt( mean_i ||Gamma_pred_i - Gamma_true_i||_2^2 )
#
#     RMSE_sigma =
#       sqrt( mean_i (sigma_pred_i - sigma_true_i)^2 )
#
# We do NOT use relative L2 as the primary plot metric. The three
# variables have different physical scales and units, so a separate
# dimensional RMSE panel for each state quantity is more interpretable.
# This is also consistent with learned-simulator literature, where
# one-step MSE/RMSE and separate rollout errors are reported rather than
# forcing heterogeneous state variables into one scalar relative norm.
# =====================================================================

@torch.inference_mode()
def one_step_error_records():
    records = []

    for pid in test_ids:
        pid = int(pid)

        batch = build_batch(
            pid,
            max_particles=MAX_INPUT_PARTICLES,
            max_queries=MAX_METRIC_QUERY_POINTS,
        )

        delta_pred, _ = predict(batch)

        state_pred, state_true, _ = (
            reconstruct_next_state_from_residual(
                batch,
                delta_pred[0],
            )
        )

        pred_np = (
            state_pred
            .cpu()
            .numpy()
        )

        true_np = (
            state_true
            .cpu()
            .numpy()
        )

        pos_diff = (
            pred_np[:, :3]
            - true_np[:, :3]
        )

        circ_diff = (
            pred_np[:, 3:6]
            - true_np[:, 3:6]
        )

        sigma_diff = (
            pred_np[:, 6]
            - true_np[:, 6]
        )

        records.append(
            {
                "pair_id": pid,
                "phase": float(
                    pair_context_map[pid].get(
                        "phase_t",
                        0.0,
                    )
                ),
                "aoa": int(
                    round(
                        float(
                            pair_context_map[pid].get(
                                "aoa_deg",
                                0.0,
                            )
                        )
                    )
                ),
                "position_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                pos_diff ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "circulation_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                circ_diff ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "sigma_rmse": float(
                    np.sqrt(
                        np.mean(
                            sigma_diff ** 2
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(records)


df_one_step = one_step_error_records()

PHASE_MIN_PLOT = 0.1
AOA_TO_PLOT = (21, 32)

one_step_specs = [
    (
        "position_rmse",
        "Position RMSE (m)",
    ),
    (
        "circulation_rmse",
        "Circulation RMSE",
    ),
    (
        "sigma_rmse",
        "Sigma RMSE",
    ),
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.2, 2.7),
    sharex=True,
    constrained_layout=True,
)

for ax, (metric, ylabel) in zip(
    axes,
    one_step_specs,
):
    for aoa, marker in zip(
        AOA_TO_PLOT,
        ("o", "s"),
    ):
        df = (
            df_one_step[
                (df_one_step["aoa"] == aoa)
                & (
                    df_one_step["phase"]
                    >= PHASE_MIN_PLOT
                )
            ]
            .sort_values("phase")
        )

        if df.empty:
            continue

        ax.plot(
            df["phase"].to_numpy(),
            df[metric].to_numpy(),
            marker=marker,
            linewidth=1.1,
            markersize=2.7,
            markerfacecolor="white",
            label=fr"AoA = {aoa}$^\circ$",
        )

    ax.set_xlabel(
        "Phase",
        fontsize=8,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=8,
    )

    ax.grid(
        True,
        linewidth=0.45,
        alpha=0.3,
        linestyle="--",
    )

    ax.tick_params(
        labelsize=7,
    )

axes[0].legend(
    frameon=False,
    fontsize=7,
    loc="best",
)

fig.suptitle(
    "One-step transition accuracy",
    fontsize=10,
)

one_step_output = (
    PLOT_DIR
    / "fig_A_one_step_RMSE.pdf"
)

fig.savefig(
    one_step_output,
    bbox_inches="tight",
)

plt.close(fig)

print(
    f"Saved Figure A: {one_step_output}"
)


# =====================================================================
# 10. FIGURE B — AUTOREGRESSIVE LONG-HORIZON ROLLOUT
# =====================================================================
#
# The rollout is genuinely autoregressive:
#
#       s_hat_{k+1} = F_theta(s_hat_k)
#
# after the first step. The next model input is constructed from the
# previously predicted state and the previous predicted field variables,
# following the same recurrence used by the training notebook's
# multi-step rollout loss.
#
# The first model call starts from a TRUE test state. After that no true
# state is fed back into the prediction loop. Ground truth is used ONLY
# to compute the error after each predicted step.
#
# We report:
#   (i)   median RMSE across rollout starts,
#   (ii)  interquartile range as a shaded band,
#   (iii) final-step RMSE,
#   (iv) time-averaged ("cumulative") RMSE over the horizon.
#
# This follows the presentation used in learned simulators and neural
# operator long-horizon studies: one-step accuracy and autoregressive
# error accumulation are treated as separate experiments.
# =====================================================================

if int(model.lookback_steps) != 0:
    raise RuntimeError(
        "This analysis rollout currently assumes lookback_steps=0, "
        f"but the loaded checkpoint has lookback_steps={model.lookback_steps}."
    )


ROLL_OUT_STEPS = 100

# Multiple starting phases are used so Figure B is not based on a single
# arbitrarily chosen initial condition. These phases are kept early
# enough that a 100-step rollout remains inside the stored trajectory.
ROLL_OUT_START_PHASES = (
    0.10,
    0.20,
    0.30,
    0.40,
)

ROLL_OUT_AOAS = (21, 32)


def rollout_frame_index_for_pair(
    pair_id: int,
) -> Tuple[str, int]:
    pair_id = int(pair_id)

    ctx = pair_context_map[pair_id]

    case = str(
        ctx.get(
            "case",
            "",
        )
    )

    frame_id = str(
        ctx.get(
            "frame_t",
            ctx.get(
                "frame_id",
                "",
            ),
        )
    ).zfill(6)

    case_idx = case_to_rollout_index.get(
        case
    )

    if case_idx is None:
        raise RuntimeError(
            f"No rollout_true_states entry for case {case!r}."
        )

    frame_map = case_to_frame_index.get(
        case,
        {},
    )

    start_idx = frame_map.get(
        frame_id
    )

    if start_idx is None:
        raise RuntimeError(
            f"Could not map case={case!r}, frame={frame_id} "
            "to rollout_true_states."
        )

    return case, int(start_idx)


def get_rollout_targets(
    pair_id: int,
    steps: int,
):
    case, start_idx = (
        rollout_frame_index_for_pair(
            pair_id
        )
    )

    case_idx = case_to_rollout_index[
        case
    ]

    states = np.asarray(
        rollout_true_states[
            case_idx
        ],
        dtype=np.float32,
    )

    if states.ndim != 3:
        raise RuntimeError(
            f"rollout_true_states[{case_idx}] has shape "
            f"{states.shape}; expected (T,N,7)."
        )

    max_available = max(
        int(states.shape[0])
        - start_idx
        - 1,
        0,
    )

    actual_steps = min(
        int(steps),
        max_available,
    )

    if actual_steps <= 0:
        return None

    targets = (
        states[
            start_idx + 1:
            start_idx + 1 + actual_steps,
            :MAX_INPUT_PARTICLES,
            :7,
        ]
    )

    phases = None

    if (
        len(rollout_phases)
        > case_idx
    ):
        phases_all = np.asarray(
            rollout_phases[
                case_idx
            ],
            dtype=np.float32,
        )

        phases = phases_all[
            start_idx + 1:
            start_idx + 1 + actual_steps
        ]

    return (
        targets,
        phases,
    )


def select_rollout_starts(
    aoa: int,
    desired_phases: Sequence[float],
    max_steps: int,
):
    candidates = []

    for pid in test_ids:
        pid = int(pid)

        ctx = pair_context_map[pid]

        pid_aoa = int(
            round(
                float(
                    ctx.get(
                        "aoa_deg",
                        0.0,
                    )
                )
            )
        )

        if pid_aoa != int(aoa):
            continue

        try:
            targets = get_rollout_targets(
                pid,
                max_steps,
            )
        except RuntimeError:
            continue

        if targets is None:
            continue

        available_steps = targets[0].shape[0]

        if available_steps < max_steps:
            continue

        candidates.append(
            (
                pid,
                float(
                    ctx.get(
                        "phase_t",
                        0.0,
                    )
                ),
            )
        )

    if not candidates:
        raise RuntimeError(
            f"No eligible autoregressive rollout starts for AoA={aoa}° "
            f"and horizon={max_steps}."
        )

    selected = []
    used = set()

    for requested_phase in desired_phases:
        remaining = [
            item
            for item in candidates
            if item[0] not in used
        ]

        if not remaining:
            break

        pid, actual_phase = min(
            remaining,
            key=lambda item: (
                abs(
                    item[1]
                    - float(requested_phase)
                ),
                item[0],
            ),
        )

        selected.append(
            (
                pid,
                actual_phase,
                float(requested_phase),
            )
        )

        used.add(pid)

    return selected



def choose_phase_pairs(
    aoa: int,
    desired_phases: Sequence[float],
):
    """
    Select the nearest distinct test pair to each requested phase at
    the requested AoA.
    """
    candidates = []

    for pid in test_ids:
        pid = int(pid)
        ctx = pair_context_map[pid]

        pid_aoa = int(
            round(
                float(
                    ctx.get("aoa_deg", 0.0)
                )
            )
        )

        if pid_aoa != int(aoa):
            continue

        candidates.append(
            (
                pid,
                float(
                    ctx.get("phase_t", 0.0)
                ),
            )
        )

    if not candidates:
        raise RuntimeError(
            f"No test pairs found for AoA={aoa}°."
        )

    selected = []
    used = set()

    for requested_phase in desired_phases:
        remaining = [
            item
            for item in candidates
            if item[0] not in used
        ]

        if not remaining:
            raise RuntimeError(
                f"Could not select distinct pairs for all "
                f"requested phases at AoA={aoa}°."
            )

        pid, actual_phase = min(
            remaining,
            key=lambda item: (
                abs(
                    item[1] - float(requested_phase)
                ),
                item[0],
            ),
        )

        selected.append(
            (
                pid,
                actual_phase,
                float(requested_phase),
            )
        )
        used.add(pid)

    for pid, actual_phase, requested_phase in selected:
        print(
            f"AoA={aoa}°: requested phase={requested_phase:.3f} "
            f"-> pair_id={pid}, actual phase={actual_phase:.6f}"
        )

    return selected


def build_rollout_x(
    template_x: torch.Tensor,
    state_phys: torch.Tensor,
    velocity_phys: torch.Tensor,
    gradU_phys: torch.Tensor,
):
    """
    Construct the normalized model input at the new autoregressive state.

    Static/non-dynamic input channels are inherited from the initial
    template, while position, circulation, sigma, velocity and gradients
    are overwritten by the autoregressive state/field.

    This reproduces the training notebook's rollout input construction.
    """
    B, N, _ = template_x.shape

    input_mean_full = torch.tensor(
        input_mean,
        dtype=template_x.dtype,
        device=template_x.device,
    ).view(
        1,
        1,
        -1,
    )

    input_std_full = torch.tensor(
        input_std,
        dtype=template_x.dtype,
        device=template_x.device,
    ).view(
        1,
        1,
        -1,
    )

    old_active_phys = (
        template_x
        * input_std_full
        + input_mean_full
    )

    full_raw = torch.zeros(
        (
            B,
            N,
            len(feature_names_all),
        ),
        dtype=template_x.dtype,
        device=template_x.device,
    )

    for local_i, global_i in enumerate(
        active_input_feature_indices
    ):
        full_raw[
            :,
            :,
            global_i,
        ] = old_active_phys[
            :,
            :,
            local_i,
        ]

    # state: x,y,z,Gamma_x,Gamma_y,Gamma_z,sigma
    full_raw[
        :,
        :,
        state_feature_indices,
    ] = state_phys

    # current predicted Eulerian field quantities evaluated at the current
    # particle positions.
    full_raw[
        :,
        :,
        velocity_feature_indices,
    ] = velocity_phys

    full_raw[
        :,
        :,
        gradient_feature_indices,
    ] = gradU_phys

    active = full_raw[
        :,
        :,
        active_input_feature_indices,
    ]

    normalized = (
        active
        - input_mean_full
    ) / input_std_full

    return torch.clamp(
        torch.nan_to_num(
            normalized
        ),
        -8.0,
        8.0,
    )


def update_global_phase(
    global_params: torch.Tensor,
    phase_value: Optional[float],
):
    gp = global_params.clone()

    if (
        phase_value is not None
        and "phase"
        in global_condition_channels
    ):
        phase_idx = (
            global_condition_channels.index(
                "phase"
            )
        )

        gp[
            :,
            phase_idx
        ] = float(
            phase_value
        )

    return gp


@torch.inference_mode()
def autoregressive_rollout(
    start_pair_id: int,
    max_steps: int,
):
    """
    Perform a genuine closed-loop rollout from one true initial state.

    Only step zero is initialized with the true state. From step one
    onward, model-generated state is recursively fed back.

    Returns per-step physical RMSEs and the corresponding target phases.
    """
    pair_id = int(start_pair_id)

    target_data = get_rollout_targets(
        pair_id,
        max_steps,
    )

    if target_data is None:
        return None

    target_states_np, target_phases_np = (
        target_data
    )

    if target_states_np.shape[0] < max_steps:
        return None

    initial_batch = build_batch(
        pair_id,
        max_particles=MAX_INPUT_PARTICLES,
        max_queries=MAX_METRIC_QUERY_POINTS,
    )

    state_phys = torch.from_numpy(
        initial_batch["state_phys"]
    ).unsqueeze(0).to(
        DEVICE,
        dtype=torch.float32,
    )

    velocity_phys = torch.from_numpy(
        initial_batch["velocity_phys"]
    ).unsqueeze(0).to(
        DEVICE,
        dtype=torch.float32,
    )

    gradU_phys = torch.from_numpy(
        initial_batch["gradU_phys"]
    ).unsqueeze(0).to(
        DEVICE,
        dtype=torch.float32,
    ).reshape(
        1,
        -1,
        3,
        3,
    )

    geom = initial_batch[
        "input_geom"
    ].clone()

    x = initial_batch[
        "x"
    ].clone()

    template_x = initial_batch[
        "x"
    ].clone()

    global_params = initial_batch[
        "global_params"
    ].clone()

    dt = float(
        initial_batch["dt"]
    )

    coord_min_t = torch.tensor(
        coord_min,
        dtype=torch.float32,
        device=DEVICE,
    ).view(1, 1, 3)

    coord_max_t = (
        coord_min_t
        + torch.tensor(
            coord_span,
            dtype=torch.float32,
            device=DEVICE,
        ).view(1, 1, 3)
    )

    rmse_records = []

    for k in range(max_steps):
        # The field decoder is evaluated AT THE CURRENT PARTICLE
        # POSITIONS. This provides the current u and gradU required by
        # the explicit Euler baseline and is exactly the field quantity
        # needed by the trained state transition.
        delta_pred, field_particle_pred = (
            model(
                geom,
                LATENT_QUERIES,
                geom,
                x,
                global_params,
                batch_dict=None,
            )
        )

        delta_phys = (
            denormalize_delta(
                delta_pred
            )[:, :, :7]
        )

        field_phys = (
            denormalize_field(
                field_particle_pred
            )
        )

        velocity_current = (
            field_phys[
                ...,
                :3
            ]
        )

        grad_current = (
            field_phys[
                ...,
                3:12
            ]
            .reshape(
                1,
                -1,
                3,
                3,
            )
        )

        gamma_current = (
            state_phys[:, :, 3:6]
        )

        physics_dx = (
            dt
            * velocity_current
        )

        physics_dgamma = (
            dt
            * torch.einsum(
                "bnij,bnj->bni",
                grad_current,
                gamma_current,
            )
        )

        next_state = (
            state_phys.clone()
        )

        next_state[
            :,
            :,
            :3
        ] = (
            state_phys[
                :,
                :,
                :3
            ]
            + physics_dx
            + delta_phys[
                :,
                :,
                :3
            ]
        )

        next_state[
            :,
            :,
            3:6
        ] = (
            state_phys[
                :,
                :,
                3:6
            ]
            + physics_dgamma
            + delta_phys[
                :,
                :,
                3:6
            ]
        )

        next_state[
            :,
            :,
            6
        ] = (
            state_phys[
                :,
                :,
                6
            ]
            + delta_phys[
                :,
                :,
                6
            ]
        )

        target_k = torch.from_numpy(
            target_states_np[
                k,
                :next_state.shape[1],
                :7,
            ]
        ).unsqueeze(0).to(
            DEVICE,
            dtype=torch.float32,
        )

        err = (
            next_state
            - target_k
        )

        pos_rmse = torch.sqrt(
            torch.mean(
                torch.sum(
                    err[
                        :,
                        :,
                        :3
                    ] ** 2,
                    dim=-1,
                )
            )
        )

        circ_rmse = torch.sqrt(
            torch.mean(
                torch.sum(
                    err[
                        :,
                        :,
                        3:6
                    ] ** 2,
                    dim=-1,
                )
            )
        )

        sigma_rmse = torch.sqrt(
            torch.mean(
                err[
                    :,
                    :,
                    6
                ] ** 2
            )
        )

        out_of_domain_fraction = (
            (
                (next_state[:, :, :3] < coord_min_t)
                |
                (next_state[:, :, :3] > coord_max_t)
            )
            .any(dim=-1)
            .float()
            .mean()
        )

        rmse_records.append(
            {
                "step": k + 1,
                "position_rmse": float(
                    pos_rmse.item()
                ),
                "circulation_rmse": float(
                    circ_rmse.item()
                ),
                "sigma_rmse": float(
                    sigma_rmse.item()
                ),
                "out_of_domain_fraction": float(
                    out_of_domain_fraction.item()
                ),
                "phase": (
                    float(
                        target_phases_np[k]
                    )
                    if target_phases_np is not None
                    else np.nan
                ),
            }
        )

        # Closed-loop state feedback.
        state_phys = next_state

        geom = normalize_geom_t(
            state_phys[:, :, :3]
        )

        phase_value = (
            None
            if target_phases_np is None
            else float(
                target_phases_np[k]
            )
        )

        # The training notebook carries the current predicted field
        # variables into the next input. We reproduce that recurrence.
        x = build_rollout_x(
            template_x,
            state_phys,
            velocity_current,
            grad_current.reshape(
                1,
                -1,
                9,
            ),
        )

        global_params = update_global_phase(
            global_params,
            phase_value,
        )

    return pd.DataFrame(
        rmse_records
    )


def summarize_rollout_records(
    rollout_records,
):
    all_rows = []

    for result in rollout_records:
        df = result["records"].copy()
        df["aoa"] = result["aoa"]
        df["start_pair_id"] = result[
            "start_pair_id"
        ]
        df["start_phase"] = result[
            "start_phase"
        ]
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(
        all_rows,
        ignore_index=True,
    )


rollout_results = []

for aoa in ROLL_OUT_AOAS:
    starts = select_rollout_starts(
        aoa,
        ROLL_OUT_START_PHASES,
        ROLL_OUT_STEPS,
    )

    print()
    print(
        f"AoA={aoa}° rollout starts:"
    )

    for pid, actual_phase, requested_phase in starts:
        print(
            f"   requested phase={requested_phase:.2f} "
            f"-> pair={pid}, actual phase={actual_phase:.6f}"
        )

        records = autoregressive_rollout(
            pid,
            ROLL_OUT_STEPS,
        )

        if records is None:
            print(
                f"   WARNING: rollout skipped for pair {pid}."
            )
            continue

        rollout_results.append(
            {
                "aoa": aoa,
                "start_pair_id": pid,
                "start_phase": actual_phase,
                "records": records,
            }
        )

df_rollout = summarize_rollout_records(
    rollout_results
)

if df_rollout.empty:
    raise RuntimeError(
        "No autoregressive rollout was successfully evaluated."
    )


def rollout_statistics(
    df: pd.DataFrame,
    metric: str,
):
    return (
        df.groupby(
            ["aoa", "step"],
            as_index=False,
        )[metric]
        .agg(
            median="median",
            q25=lambda x: np.percentile(x, 25),
            q75=lambda x: np.percentile(x, 75),
        )
    )


TRAINING_ROLLOUT_HORIZON = int(
    cfg.get(
        "rollout_steps_max",
        1,
    )
)

rollout_specs = [
    ("position_rmse", "Position RMSE (m)"),
    ("circulation_rmse", "Circulation RMSE"),
    ("sigma_rmse", "Sigma RMSE"),
]

# ---------------------------------------------------------------------
# Figure B1: physical RMSE versus rollout step.
# This is the standard long-horizon error-growth visualization used in
# learned-simulator work. The shaded area is the IQR over rollout starts.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.2, 2.7),
    sharex=True,
    constrained_layout=True,
)

for ax, (metric, ylabel) in zip(
    axes,
    rollout_specs,
):
    for aoa, marker in zip(
        ROLL_OUT_AOAS,
        ("o", "s"),
    ):
        stats = rollout_statistics(
            df_rollout[
                df_rollout["aoa"] == aoa
            ],
            metric,
        )

        if stats.empty:
            continue

        step = stats["step"].to_numpy()
        median = stats["median"].to_numpy()
        q25 = stats["q25"].to_numpy()
        q75 = stats["q75"].to_numpy()

        ax.plot(
            step,
            median,
            linewidth=1.2,
            label=fr"AoA = {aoa}$^\circ$",
        )

        ax.fill_between(
            step,
            q25,
            q75,
            alpha=0.16,
        )

        sampled = stats[
            stats["step"] % 10 == 0
        ]

        ax.plot(
            sampled["step"].to_numpy(),
            sampled["median"].to_numpy(),
            marker=marker,
            linestyle="None",
            markersize=2.8,
            markerfacecolor="white",
        )

    ax.axvline(
        TRAINING_ROLLOUT_HORIZON,
        linewidth=0.8,
        linestyle=":",
    )

    ax.set_xlabel(
        "Rollout step",
        fontsize=8,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=8,
    )

    ax.grid(
        True,
        linewidth=0.45,
        alpha=0.3,
        linestyle="--",
    )

    ax.tick_params(
        labelsize=7,
    )

axes[0].legend(
    frameon=False,
    fontsize=7,
    loc="best",
)

fig.suptitle(
    "Autoregressive rollout error",
    fontsize=10,
)

rollout_rmse_output = (
    PLOT_DIR
    / "fig_B1_autoregressive_rollout_RMSE.pdf"
)

fig.savefig(
    rollout_rmse_output,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure B2: error amplification E_k / E_1.
#
# This removes the physical units from the vertical axis and answers the
# specific long-horizon question: how many times larger is the error after
# k recursive steps than after the first recursive step?
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.2, 2.7),
    sharex=True,
    constrained_layout=True,
)

for ax, (metric, title) in zip(
    axes,
    rollout_specs,
):
    for aoa, marker in zip(
        ROLL_OUT_AOAS,
        ("o", "s"),
    ):
        stats = rollout_statistics(
            df_rollout[
                df_rollout["aoa"] == aoa
            ],
            metric,
        )

        if stats.empty:
            continue

        step = stats["step"].to_numpy()
        median = stats["median"].to_numpy()
        q25 = stats["q25"].to_numpy()
        q75 = stats["q75"].to_numpy()

        e1 = float(median[0])

        if e1 <= 1e-12:
            continue

        amp = median / e1
        amp_q25 = q25 / e1
        amp_q75 = q75 / e1

        ax.plot(
            step,
            amp,
            linewidth=1.2,
            label=fr"AoA = {aoa}$^\circ$",
        )

        ax.fill_between(
            step,
            amp_q25,
            amp_q75,
            alpha=0.16,
        )

        sampled = stats[
            stats["step"] % 10 == 0
        ]

        ax.plot(
            sampled["step"].to_numpy(),
            (
                sampled["median"].to_numpy()
                / e1
            ),
            marker=marker,
            linestyle="None",
            markersize=2.8,
            markerfacecolor="white",
        )

    ax.axhline(
        1.0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.axvline(
        TRAINING_ROLLOUT_HORIZON,
        linewidth=0.8,
        linestyle=":",
    )

    ax.set_yscale("log")
    ax.set_xlabel(
        "Rollout step",
        fontsize=8,
    )
    ax.set_title(
        title.replace(" RMSE", ""),
        fontsize=9,
    )
    ax.grid(
        True,
        which="both",
        linewidth=0.45,
        alpha=0.3,
        linestyle="--",
    )
    ax.tick_params(
        labelsize=7,
    )

axes[0].set_ylabel(
    r"Error amplification $E_k/E_1$",
    fontsize=8,
)

axes[0].legend(
    frameon=False,
    fontsize=7,
    loc="best",
)

fig.suptitle(
    "Autoregressive error amplification",
    fontsize=10,
)

rollout_amp_output = (
    PLOT_DIR
    / "fig_B2_autoregressive_error_amplification.pdf"
)

fig.savefig(
    rollout_amp_output,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure B3: fraction of particles leaving the normalized training
# coordinate domain. This is a diagnostic, not an accuracy metric.
# ---------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(4.4, 2.7),
    constrained_layout=True,
)

for aoa, marker in zip(
    ROLL_OUT_AOAS,
    ("o", "s"),
):
    ood = (
        df_rollout[
            df_rollout["aoa"] == aoa
        ]
        .groupby("step")[
            "out_of_domain_fraction"
        ]
        .agg(
            median="median",
            q25=lambda x: np.percentile(x, 25),
            q75=lambda x: np.percentile(x, 75),
        )
        .reset_index()
    )

    if ood.empty:
        continue

    step = ood["step"].to_numpy()

    ax.plot(
        step,
        100.0 * ood["median"].to_numpy(),
        linewidth=1.2,
        label=fr"AoA = {aoa}$^\circ$",
    )

    ax.fill_between(
        step,
        100.0 * ood["q25"].to_numpy(),
        100.0 * ood["q75"].to_numpy(),
        alpha=0.16,
    )

    sampled = ood[
        ood["step"] % 10 == 0
    ]

    ax.plot(
        sampled["step"].to_numpy(),
        100.0 * sampled["median"].to_numpy(),
        marker=marker,
        linestyle="None",
        markersize=2.8,
        markerfacecolor="white",
    )

ax.axvline(
    TRAINING_ROLLOUT_HORIZON,
    linewidth=0.8,
    linestyle=":",
)

ax.set_xlabel(
    "Rollout step",
    fontsize=8,
)

ax.set_ylabel(
    "Particles outside input domain (%)",
    fontsize=8,
)

ax.set_title(
    "Autoregressive domain validity",
    fontsize=9,
)

ax.set_ylim(
    0.0,
    100.0,
)

ax.grid(
    True,
    linewidth=0.45,
    alpha=0.3,
    linestyle="--",
)

ax.tick_params(
    labelsize=7,
)

ax.legend(
    frameon=False,
    fontsize=7,
    loc="best",
)

ood_output = (
    PLOT_DIR
    / "fig_B3_rollout_domain_validity.pdf"
)

fig.savefig(
    ood_output,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Per-start diagnostic summary.
# ---------------------------------------------------------------------
summary_rows = []

for result in rollout_results:
    df_start = result["records"]

    for metric, ylabel in rollout_specs:
        first = float(
            df_start.iloc[0][metric]
        )

        final = float(
            df_start.iloc[-1][metric]
        )

        max_value = float(
            df_start[metric].max()
        )

        mean_over_horizon = float(
            df_start[metric].mean()
        )

        summary_rows.append(
            {
                "AoA": result["aoa"],
                "start_pair_id": result[
                    "start_pair_id"
                ],
                "start_phase": result[
                    "start_phase"
                ],
                "metric": ylabel,
                "first_step_rmse": first,
                "final_step_rmse": final,
                "max_rmse": max_value,
                "time_averaged_rmse": mean_over_horizon,
                "final_to_first_amplification": (
                    final / first
                    if first > 1e-12
                    else np.nan
                ),
                "max_out_of_domain_fraction": float(
                    df_start[
                        "out_of_domain_fraction"
                    ].max()
                ),
            }
        )

rollout_summary = pd.DataFrame(
    summary_rows
)

rollout_summary.to_csv(
    PLOT_DIR
    / "fig_B_rollout_summary.csv",
    index=False,
)

print()
print("=" * 72)
print("Autoregressive rollout summary")
print("=" * 72)
print(
    rollout_summary.to_string(
        index=False
    )
)


# =====================================================================
# 11. NATIVE HDF5 FIELD-SLICE COMPARISON
# =====================================================================

def _numeric_frame_candidates(path: Path):
    """
    Extract plausible frame numbers from an HDF5 filename.

    FLOWUnsteady field files are commonly named with an fdom frame number,
    but this parser also accepts generic numeric suffixes so that minor
    filename conventions do not break the analysis.
    """
    stem = path.stem

    values = []

    for match in re.finditer(
        r"(?:^|[._-])fdom[._-]?(\d+)(?:$|[._-])",
        stem,
        flags=re.IGNORECASE,
    ):
        values.append(int(match.group(1)))

    if not values:
        values.extend(
            int(m.group(1))
            for m in re.finditer(
                r"(?:^|[._-])(\d+)(?:$|[._-])",
                stem,
            )
        )

    if not values:
        values.extend(
            int(m)
            for m in re.findall(r"\d+", stem)
        )

    return values


def _find_native_h5_for_pair(
    pair_id: int,
) -> Path:
    """
    Locate the exact native HDF5 field file for pair_id.

    The search is restricted to:
        FIELD_ROOT / case

    and requires the exact frame_t encoded in the pair metadata.
    A nearby frame is never silently substituted.
    """
    pair_id = int(pair_id)
    ctx = pair_context_map[pair_id]

    case = str(
        ctx.get(
            "case",
            "",
        )
    ).strip()

    frame_text = str(
        ctx.get(
            "frame_t",
            ctx.get(
                "frame_id",
                "",
            ),
        )
    ).strip()

    if not case:
        raise RuntimeError(
            f"pair_id={pair_id} has no case identifier."
        )

    if not frame_text:
        raise RuntimeError(
            f"pair_id={pair_id} has no frame_t/frame_id."
        )

    try:
        frame_number = int(frame_text)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid frame identifier {frame_text!r} "
            f"for pair_id={pair_id}."
        ) from exc

    case_dir = FIELD_ROOT / case

    if not case_dir.is_dir():
        raise FileNotFoundError(
            "Native HDF5 case directory does not exist:\n"
            f"  {case_dir}\n\n"
            "Configured FIELD_ROOT:\n"
            f"  {FIELD_ROOT}\n"
        )

    candidates = sorted(
        set(
            list(case_dir.rglob("*.h5"))
            + list(case_dir.rglob("*.hdf5"))
        ),
        key=lambda p: str(p),
    )

    if not candidates:
        raise FileNotFoundError(
            f"No HDF5 files were found below:\n  {case_dir}"
        )

    # First pass: use parsed fdom frame IDs.
    exact_matches = [
        path
        for path in candidates
        if frame_number in _numeric_frame_candidates(path)
    ]

    # Prefer a filename explicitly containing fdom + exact frame.
    exact_fdom = [
        path
        for path in exact_matches
        if re.search(
            rf"fdom[._-]?0*{frame_number}(?:[._-]|$)",
            path.stem,
            flags=re.IGNORECASE,
        )
    ]

    search_order = exact_fdom + [
        path
        for path in exact_matches
        if path not in exact_fdom
    ]

    # Verify the candidate actually contains the required datasets.
    for path in search_order:
        try:
            with h5py.File(path, "r") as h5:
                if "nodes" in h5 and "U" in h5:
                    return path
        except OSError:
            continue

    available = sorted(
        {
            value
            for path in candidates
            for value in _numeric_frame_candidates(path)
        }
    )

    raise FileNotFoundError(
        f"No exact native HDF5 field with frame={frame_number} "
        f"was found for case={case!r}.\n"
        f"Searched recursively below:\n  {case_dir}\n"
        f"Available parsed frame IDs (first 40): {available[:40]}"
    )


def _load_native_field(
    pair_id: int,
):
    """
    Load the exact physical native VPM field from HDF5.
    """
    path = _find_native_h5_for_pair(pair_id)
    ctx = pair_context_map[int(pair_id)]

    with h5py.File(path, "r") as h5:
        nodes = np.asarray(
            h5["nodes"][()],
            dtype=np.float32,
        )
        velocity = np.asarray(
            h5["U"][()],
            dtype=np.float32,
        )

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(
            f"{path}: nodes has shape {nodes.shape}; expected (N, 3)."
        )

    # Be robust to either (N,3) or (3,N) velocity storage.
    if velocity.shape == nodes.shape:
        pass
    elif velocity.ndim == 2 and velocity.T.shape == nodes.shape:
        velocity = velocity.T
    else:
        raise ValueError(
            f"{path}: U has shape {velocity.shape}; "
            f"expected {nodes.shape} or {nodes.shape[::-1]}."
        )

    freestream = float(
        np.linalg.norm(
            np.asarray(
                ctx.get(
                    "freestream",
                    [1.0, 0.0, 0.0],
                ),
                dtype=np.float64,
            )
        )
    )

    if freestream <= 0.0:
        raise RuntimeError(
            f"pair_id={pair_id} has non-positive freestream magnitude."
        )

    return (
        path,
        nodes,
        velocity,
        freestream,
    )


def native_true_slice_on_common_grid(
    pair_id: int,
    y_slice: float,
    resolution: int = 256,
    y_tolerance: float = 0.01,
):
    """
    Build the true |u|/U_inf field on a regular x-z grid obtained ONLY
    from the exact native HDF5 frame.

    GINO is subsequently queried on this exact same x-z grid.
    """
    path, nodes, velocity, freestream = (
        _load_native_field(pair_id)
    )

    y = nodes[:, 1]
    y_distance = np.abs(
        y - float(y_slice)
    )

    mask = (
        y_distance <= float(y_tolerance)
    )

    if int(mask.sum()) < 100:
        # The physical plane is fixed; only the extraction tolerance is
        # relaxed enough to recover a populated native slice.
        nearest_distance = float(
            y_distance.min()
        )

        mask = (
            y_distance
            <= max(
                float(y_tolerance),
                2.0 * nearest_distance,
            )
        )

    if int(mask.sum()) < 100:
        raise RuntimeError(
            f"{path}: only {int(mask.sum())} native points were found "
            f"near y={y_slice:.8f}."
        )

    x = nodes[mask, 0]
    z = nodes[mask, 2]

    u_mag_nd = (
        np.linalg.norm(
            velocity[mask],
            axis=1,
        )
        / freestream
    )

    x_lo = float(x.min())
    x_hi = float(x.max())
    z_lo = float(z.min())
    z_hi = float(z.max())

    if not (
        x_hi > x_lo
        and z_hi > z_lo
    ):
        raise RuntimeError(
            f"{path}: degenerate x-z native slice."
        )

    x_vals = np.linspace(
        x_lo,
        x_hi,
        int(resolution),
        dtype=np.float32,
    )

    z_vals = np.linspace(
        z_lo,
        z_hi,
        int(resolution),
        dtype=np.float32,
    )

    XX, ZZ = np.meshgrid(
        x_vals,
        z_vals,
        indexing="xy",
    )

    true_grid = griddata(
        (x, z),
        u_mag_nd,
        (XX, ZZ),
        method="linear",
        fill_value=np.nan,
    )

    nan_mask = np.isnan(true_grid)

    if nan_mask.any():
        nearest_grid = griddata(
            (x, z),
            u_mag_nd,
            (XX, ZZ),
            method="nearest",
    )
        true_grid[nan_mask] = (
            nearest_grid[nan_mask]
        )

    ctx = pair_context_map[int(pair_id)]

    print(
        f"   HDF5 true field: {path}"
    )
    print(
        f"   Exact frame: "
        f"{int(str(ctx.get('frame_t', '0')))}"
    )
    print(
        f"   Native points near y={y_slice:.8f}: "
        f"{int(mask.sum())}"
    )
    print(
        f"   Common grid: "
        f"x=[{x_lo:.6f}, {x_hi:.6f}], "
        f"z=[{z_lo:.6f}, {z_hi:.6f}]"
    )

    return (
        x_vals,
        z_vals,
        true_grid.astype(np.float32),
        freestream,
    )


@torch.inference_mode()
def predict_dense_slice(
    pair_id: int,
    x_vals: np.ndarray,
    z_vals: np.ndarray,
    y_slice: float,
    query_chunk_size: int = 32768,
):
    """
    Evaluate the GINO field decoder on exactly the supplied x-z grid.

    The GNO/FNO encoding is performed once. The field decoder is then
    evaluated in chunks so that a 256x256 visualization grid does not
    create an unnecessary peak-memory spike.
    """
    batch = build_batch(
        pair_id,
        max_particles=MAX_INPUT_PARTICLES,
        max_queries=MAX_METRIC_QUERY_POINTS,
    )

    x_vals = np.asarray(
        x_vals,
        dtype=np.float32,
    )

    z_vals = np.asarray(
        z_vals,
        dtype=np.float32,
    )

    XX, ZZ = np.meshgrid(
        x_vals,
        z_vals,
        indexing="xy",
    )

    YY = np.full_like(
        XX,
        float(y_slice),
        dtype=np.float32,
    )

    queries = np.stack(
        [
            XX.ravel(),
            YY.ravel(),
            ZZ.ravel(),
        ],
        axis=-1,
    ).astype(np.float32)

    # Encode/process once; only the field-decoder query is chunked.
    base_latent, grids, _, _ = model.encode_process(
        batch["input_geom"],
        LATENT_QUERIES,
        batch["x"],
        batch["global_params"],
    )

    del base_latent

    grid = grids[0]

    field_chunks = []

    chunk_size = max(
        int(query_chunk_size),
        1,
    )

    for start in range(
        0,
        queries.shape[0],
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            queries.shape[0],
        )

        q_chunk_np = queries[
            start:stop
        ]

        q_chunk = torch.from_numpy(
            normalize_xyz(
                q_chunk_np
            )
        ).to(
            DEVICE,
            dtype=torch.float32,
        )

        sampled = model.sample_grid(
            grid,
            q_chunk,
        )

        field_chunk = model.field_decoder(
            torch.cat(
                [
                    sampled,
                    positional_encoding(
                        q_chunk.unsqueeze(0),
                        model.query_pe_freqs,
                    ).squeeze(0),
                ],
                dim=-1,
            )
        )

        velocity_chunk = (
            field_chunk[:, :3]
            * field_std_t[:3]
            + field_mean_t[:3]
        )

        magnitude_chunk = (
            torch.linalg.norm(
                velocity_chunk,
                dim=1,
            )
            .cpu()
            .numpy()
        )

        field_chunks.append(
            magnitude_chunk.astype(
                np.float32
            )
        )

        del (
            q_chunk,
            sampled,
            field_chunk,
            velocity_chunk,
        )

    magnitude = np.concatenate(
        field_chunks,
        axis=0,
    ).reshape(
        len(z_vals),
        len(x_vals),
    )

    del (
        grids,
        grid,
        field_chunks,
    )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return (
        XX,
        ZZ,
        magnitude,
    )



def plot_field_slice_figure(
    aoa: int,
    selected_phases: Sequence[float],
    profile_x_values: Sequence[float],
    z_clip_max: float,
    field_resolution: int = 256,
):
    if len(selected_phases) != len(
        profile_x_values
    ):
        raise ValueError(
            "selected_phases and profile_x_values "
            "must have equal length."
        )

    selections = choose_phase_pairs(
        aoa,
        selected_phases,
    )

    if not selections:
        raise RuntimeError(
            f"No field-slice selections for AoA={aoa}°."
        )

    rows = []

    for (
        target_phase,
        profile_x,
        selected,
    ) in zip(
        selected_phases,
        profile_x_values,
        selections,
    ):
        pid, actual_phase, requested_phase = (
            selected
        )

        y_val = Y_PLANE

        # TRUE physical field from the exact native HDF5 frame.
        (
            x_grid,
            z_grid,
            true_mag,
            freestream,
        ) = native_true_slice_on_common_grid(
            pid,
            y_slice=y_val,
            resolution=field_resolution,
        )

        # GINO is evaluated on exactly the same grid.
        XX, ZZ, pred_mag = (
            predict_dense_slice(
                pid,
                x_vals=x_grid,
                z_vals=z_grid,
                y_slice=y_val,
            )
        )

        pred_mag_nd = (
            pred_mag
            / freestream
        )

        vmax = 1.05 * max(
            float(
                np.nanmax(
                    true_mag
                )
            ),
            float(
                np.nanmax(
                    pred_mag_nd
                )
            ),
        )

        rows.append(
            {
                "pid": pid,
                "requested_phase": float(
                    requested_phase
                ),
                "actual_phase": float(
                    actual_phase
                ),
                "profile_x": float(
                    profile_x
                ),
                "x_grid": x_grid,
                "z_grid": z_grid,
                "XX": XX,
                "ZZ": ZZ,
                "true": true_mag,
                "pred": pred_mag_nd,
                "vmax": max(
                    vmax,
                    1e-8,
                ),
            }
        )

    fig = plt.figure(
        figsize=(
            7.5,
            1.9 * len(rows),
        )
    )

    gs = fig.add_gridspec(
        len(rows),
        4,
        width_ratios=[
            1.0,
            1.0,
            0.045,
            1.05,
        ],
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=0.07,
        wspace=0.22,
        hspace=0.32,
    )

    for r, row in enumerate(rows):
        ax_true = fig.add_subplot(
            gs[r, 0]
        )

        ax_pred = fig.add_subplot(
            gs[r, 1]
        )

        ax_cb = fig.add_subplot(
            gs[r, 2]
        )

        ax_prof = fig.add_subplot(
            gs[r, 3]
        )

        # Both panels now use the EXACT SAME physical domain.
        xlim = (
            float(row["x_grid"].min()),
            float(row["x_grid"].max()),
        )

        ylim = (
            float(row["z_grid"].min()),
            float(row["z_grid"].max()),
        )

        im = ax_true.pcolormesh(
            row["XX"],
            row["ZZ"],
            row["true"],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=row["vmax"],
            rasterized=True,
        )

        ax_true.set_xlim(*xlim)
        ax_true.set_ylim(*ylim)
        ax_true.set_aspect(
            "equal",
            adjustable="box",
        )

        ax_true.set_xticks([])
        ax_true.set_yticks([])
        ax_true.grid(False)

        ax_true.set_ylabel(
            f"phase={row['actual_phase']:.2f}",
            fontsize=8,
        )

        if r == 0:
            ax_true.set_title(
                "True $|u|/U_\\infty$",
                fontsize=9,
                pad=3,
            )

        for sp in ax_true.spines.values():
            sp.set_linewidth(0.6)

        ax_pred.pcolormesh(
            row["XX"],
            row["ZZ"],
            row["pred"],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=row["vmax"],
            rasterized=True,
        )

        ax_pred.set_xlim(*xlim)
        ax_pred.set_ylim(*ylim)
        ax_pred.set_aspect(
            "equal",
            adjustable="box",
        )

        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.grid(False)

        if r == 0:
            ax_pred.set_title(
                "Predicted $|u|/U_\\infty$",
                fontsize=9,
                pad=3,
            )

        for sp in ax_pred.spines.values():
            sp.set_linewidth(0.6)

        cb = fig.colorbar(
            im,
            cax=ax_cb,
        )

        cb.ax.tick_params(
            labelsize=6,
            length=2,
            width=0.6,
        )

        cb.outline.set_linewidth(0.6)

        if r == 0:
            cb.set_label(
                "$|u|/U_\\infty$",
                fontsize=8,
                labelpad=2,
            )

        x0 = row["profile_x"]

        ix = int(
            np.argmin(
                np.abs(
                    row["x_grid"] - x0
                )
            )
        )

        z_grid = row["z_grid"]

        true_profile = row["true"][
            :,
            ix,
        ]

        pred_profile = row["pred"][
            :,
            ix,
        ]

        mask_z = (
            z_grid <= float(
                z_clip_max
            )
        )

        ax_prof.plot(
            true_profile[mask_z],
            z_grid[mask_z],
            "-",
            linewidth=1.2,
            label="True",
        )

        ax_prof.plot(
            pred_profile[mask_z],
            z_grid[mask_z],
            "--",
            linewidth=1.2,
            label="Predicted",
        )

        ax_prof.set_xlabel(
            "$|u|/U_\\infty$",
            fontsize=8,
        )

        ax_prof.set_ylabel(
            "z",
            fontsize=8,
        )

        ax_prof.set_ylim(
            float(z_grid.min()),
            min(
                float(
                    z_clip_max
                ),
                float(
                    z_grid.max()
                ),
            ),
        )

        ax_prof.tick_params(
            labelsize=7
        )

        ax_prof.text(
            0.97,
            0.05,
            f"x = {x0:.2f}",
            transform=ax_prof.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )

        if r == 0:
            ax_prof.set_title(
                "Profiles",
                fontsize=9,
                pad=3,
            )

            ax_prof.legend(
                frameon=False,
                fontsize=7,
                loc="best",
                handlelength=1.8,
                handletextpad=0.4,
            )

    output = (
        PLOT_DIR
        / f"fig_field_slice_AoA{aoa}.pdf"
    )

    fig.savefig(
        output,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved field figure: {output}"
    )


# =====================================================================
# 12. GENERATE FINAL FIGURES
# =====================================================================

print()
print(
    "Generating AoA = 21° native-HDF5 field comparison ..."
)

plot_field_slice_figure(
    aoa=21,
    selected_phases=SELECTED_PHASES,
    profile_x_values=PROFILE_X_VALUES,
    z_clip_max=Z_CLIP_AOA_21,
    field_resolution=256,
)

print()
print(
    "Generating AoA = 32° native-HDF5 field comparison ..."
)

plot_field_slice_figure(
    aoa=32,
    selected_phases=SELECTED_PHASES,
    profile_x_values=PROFILE_X_VALUES,
    z_clip_max=Z_CLIP_AOA_32,
    field_resolution=256,
)

print()
print("=" * 72)
print("GINO evaluation complete.")
print(
    f"Figure A: {PLOT_DIR / 'fig_A_one_step_RMSE.pdf'}"
)
print(
    f"Figure B: {PLOT_DIR / 'fig_B1_autoregressive_rollout_RMSE.pdf'}"
)
print(
    f"Field 21: {PLOT_DIR / 'fig_field_slice_AoA21.pdf'}"
)
print(
    f"Field 32: {PLOT_DIR / 'fig_field_slice_AoA32.pdf'}"
)
print(
    f"Rollout summary: "
    f"{PLOT_DIR / 'fig_B_rollout_summary.csv'}"
)
print("=" * 72)
