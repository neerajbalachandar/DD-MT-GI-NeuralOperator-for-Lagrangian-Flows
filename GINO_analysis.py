import inspect
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

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


CHECKPOINT_RUN_TAG = "gino_evolution_sharedlatent_4"
SPANWISE_SLICE_FRACTION = 0.5
FIELD_FIGURE_PHASES = (0.3, 0.5, 0.7)
FIELD_FIGURE_PROFILE_STATIONS = (0.20, 0.40, 0.60)
PROFILE_Z_CLIP_AOA_21 = 2.0
PROFILE_Z_CLIP_AOA_32 = 2.5
MAX_MODEL_INPUT_PARTICLES = 4096
MAX_MODEL_QUERY_POINTS = 4096
DEFAULT_NATIVE_FIELD_ROOT = "/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task2"
RANDOM_SEED = 42
ROLLOUT_HORIZON = 100
ROLLOUT_START_PHASES = (0.10, 0.20, 0.30, 0.40)
ROLLOUT_ANGLES_OF_ATTACK = (21, 32)
MIN_PHASE_TO_PLOT = 0.1
ONE_STEP_ANGLES_OF_ATTACK = (21, 32)


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def locate_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        dataset_candidates = (
            candidate / "Processed_Data" / "processed_data_split3",
            candidate / "processed_data_split3",
        )
        if any(p.is_dir() for p in dataset_candidates):
            return candidate
    return cwd


PROJECT_ROOT = locate_project_root()
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset_file() -> Path:
    env = os.environ.get("GINO_DATASET", "").strip()
    if env:
        path = Path(env).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"GINO_DATASET does not exist: {path}")
        return path

    candidates = [
        PROJECT_ROOT / "Processed_Data" / "processed_data_split3" / "particle_evolution_dataset.npz",
        PROJECT_ROOT / "processed_data_split3" / "particle_evolution_dataset.npz",
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return existing[0]
    raise FileNotFoundError(
        "Could not find particle_evolution_dataset.npz. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def find_checkpoint_file() -> Path:
    env = os.environ.get("GINO_CHECKPOINT", "").strip()
    if env:
        path = Path(env).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"GINO_CHECKPOINT does not exist: {path}")
        return path

    exact_training = PROJECT_ROOT / "pt_model" / "sharedlatent_GINO_4" / f"{CHECKPOINT_RUN_TAG}_best_model.pt"
    uppercase_fallback = PROJECT_ROOT / "Pt_Model" / "sharedlatent_GINO_4" / f"{CHECKPOINT_RUN_TAG}_best_model.pt"

    if exact_training.exists():
        return exact_training
    if uppercase_fallback.exists():
        print("WARNING: using uppercase Pt_Model fallback:", uppercase_fallback)
        return uppercase_fallback

    matches = list(PROJECT_ROOT.rglob(f"{CHECKPOINT_RUN_TAG}_best_model.pt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple matching checkpoints found. Set GINO_CHECKPOINT explicitly:\n"
            + "\n".join(str(p) for p in matches)
        )
    raise FileNotFoundError(
        f"Could not find checkpoint. Expected:\n  {exact_training}\nor:\n  {uppercase_fallback}"
    )


DATASET_FILE = find_dataset_file()
CHECKPOINT_FILE = find_checkpoint_file()
NATIVE_FIELD_ROOT = Path(os.environ.get("GINO_FIELD_ROOT", DEFAULT_NATIVE_FIELD_ROOT)).expanduser()

print("=" * 72)
print(f"Device          : {DEVICE}")
print(f"Repository      : {PROJECT_ROOT}")
print(f"Checkpoint      : {CHECKPOINT_FILE}")
print(f"Dataset         : {DATASET_FILE}")
print(f"Native fields   : {NATIVE_FIELD_ROOT}")
print("=" * 72)


checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
model_config = dict(checkpoint["config"])

if "checkpoint_tag" in checkpoint and checkpoint["checkpoint_tag"] != CHECKPOINT_RUN_TAG:
    raise RuntimeError(
        f"Checkpoint tag mismatch: expected {CHECKPOINT_RUN_TAG!r}, got {checkpoint['checkpoint_tag']!r}."
    )

dataset = np.load(DATASET_FILE, allow_pickle=True)

all_feature_names = [str(x) for x in np.asarray(dataset["feature_names"]).tolist()]
model_input_features = [str(x) for x in checkpoint["feature_names"]]
delta_target_names = [str(x) for x in checkpoint["target_names"]]
field_output_names = [str(x) for x in checkpoint["field_target_names"]]

if len(field_output_names) < 3 or field_output_names[:3] != ["u_x", "u_y", "u_z"]:
    raise RuntimeError(f"Field targets must begin with u_x, u_y, u_z; got {field_output_names}")

required_features = [
    "x", "y", "z",
    "u_x", "u_y", "u_z",
    "Gamma_x", "Gamma_y", "Gamma_z",
    "sigma",
    "gradU_xx", "gradU_xy", "gradU_xz",
    "gradU_yx", "gradU_yy", "gradU_yz",
    "gradU_zx", "gradU_zy", "gradU_zz",
]
missing_features = [n for n in required_features if n not in all_feature_names]
if missing_features:
    raise KeyError("Dataset missing required feature channels: " + ", ".join(missing_features))

model_input_indices = [all_feature_names.index(n) for n in model_input_features]
position_indices = [all_feature_names.index(n) for n in ("x", "y", "z")]
velocity_indices = [all_feature_names.index(n) for n in ("u_x", "u_y", "u_z")]
velocity_gradient_indices = [all_feature_names.index(f"gradU_{i}{j}") for i in "xyz" for j in "xyz"]
particle_state_indices = [
    all_feature_names.index(n)
    for n in ("x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma")
]

all_input_features = np.asarray(dataset["inputs_t"], dtype=np.float32)
normalized_delta_targets = np.asarray(
    dataset["targets_residual_norm" if "targets_residual_norm" in dataset.files else "targets_delta_norm"],
    dtype=np.float32,
)
physical_delta_targets = np.asarray(dataset["targets_delta"], dtype=np.float32)
all_field_query_coords = np.asarray(dataset["query_coords"], dtype=np.float32)
normalized_field_targets = np.asarray(dataset["targets_velocity_field_norm"], dtype=np.float32)
field_query_mask = np.asarray(dataset["field_query_mask"], dtype=bool)

pair_frame_ranges = list(
    dataset["pair_ranges"] if "pair_ranges" in dataset.files else dataset["frame_ranges"]
)
pair_metadata = list(
    dataset["pair_contexts"] if "pair_contexts" in dataset.files else dataset["frame_contexts"]
)
test_pair_ids = np.asarray(
    dataset["test_pair_ids"] if "test_pair_ids" in dataset.files else [], dtype=np.int64
)

feature_mean = np.asarray(checkpoint["input_mean"], dtype=np.float32).reshape(-1)
feature_std = np.maximum(
    np.asarray(checkpoint["input_std"], dtype=np.float32).reshape(-1), 1e-8
)
delta_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32).reshape(-1)
delta_std = np.maximum(
    np.asarray(checkpoint["target_std"], dtype=np.float32).reshape(-1), 1e-8
)
field_mean = np.asarray(checkpoint["field_mean"], dtype=np.float32).reshape(-1)
field_std = np.maximum(
    np.asarray(checkpoint["field_std"], dtype=np.float32).reshape(-1), 1e-8
)
position_min = np.asarray(checkpoint["coord_min"], dtype=np.float32).reshape(3)
position_span = np.maximum(
    np.asarray(checkpoint["coord_span"], dtype=np.float32).reshape(3), 1e-8
)

slice_y_mid = float(position_min[1] + 0.5 * position_span[1])
slice_y_half_span = float(0.5 * position_span[1])
SLICE_Y = float(slice_y_mid + SPANWISE_SLICE_FRACTION * slice_y_half_span)
print(f"Validated field slice: y = {SLICE_Y:.8f} (fraction = {SPANWISE_SLICE_FRACTION})")

global_param_names = list(
    checkpoint.get("global_condition_channels", model_config["global_condition_channels"])
)
num_delta_channels = len(delta_target_names)
num_field_channels = len(field_output_names)

if len(feature_mean) != len(model_input_features) or len(feature_std) != len(model_input_features):
    raise RuntimeError(
        f"Input normalization length does not match {len(model_input_features)} input features."
    )
if len(delta_mean) < num_delta_channels or len(delta_std) < num_delta_channels:
    raise RuntimeError("Delta normalization shorter than delta output dimension.")
if len(field_mean) != num_field_channels or len(field_std) != num_field_channels:
    raise RuntimeError("Field normalization dimension does not match field output channels.")

feature_mean_t = torch.tensor(feature_mean, dtype=torch.float32, device=DEVICE)
feature_std_t = torch.tensor(feature_std, dtype=torch.float32, device=DEVICE)
delta_mean_t = torch.tensor(delta_mean[:num_delta_channels], dtype=torch.float32, device=DEVICE)
delta_std_t = torch.tensor(delta_std[:num_delta_channels], dtype=torch.float32, device=DEVICE)
field_mean_t = torch.tensor(field_mean, dtype=torch.float32, device=DEVICE)
field_std_t = torch.tensor(field_std, dtype=torch.float32, device=DEVICE)


def to_context_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        item = obj.item()
        if isinstance(item, dict):
            return item
    return dict(obj)


pair_metadata_map = {
    int(i): to_context_dict(pair_metadata[int(i)]) for i in range(len(pair_metadata))
}

case_min_particle_count = {}
for pair_id in range(len(pair_frame_ranges)):
    case = str(pair_metadata_map[pair_id].get("case", ""))
    n = int(pair_frame_ranges[pair_id][5])
    case_min_particle_count[case] = min(case_min_particle_count.get(case, n), n)

print(f"Input features  : {model_input_features}")
print(f"Delta targets   : {delta_target_names}")
print(f"Field targets   : {field_output_names}")
print(f"Global channels : {global_param_names}")
print(f"Test pairs      : {len(test_pair_ids)}")
print(f"Case n_min      : {case_min_particle_count}")


rollout_case_names = [
    str(x) for x in np.asarray(
        dataset["rollout_cases"] if "rollout_cases" in dataset.files else []
    ).tolist()
]
rollout_ground_truth = list(
    dataset["rollout_true_states"] if "rollout_true_states" in dataset.files else []
)
rollout_phase_values = list(
    dataset["rollout_phases"] if "rollout_phases" in dataset.files else []
)

case_to_rollout_idx = {case: i for i, case in enumerate(rollout_case_names)}

case_to_frame_ids_from_pairs = defaultdict(list)
for pair_id, ctx in pair_metadata_map.items():
    case = str(ctx.get("case", ""))
    if not case:
        continue
    frame_id = str(ctx.get("frame_t", ctx.get("frame_id", ""))).zfill(6)
    case_to_frame_ids_from_pairs[case].append(frame_id)

case_frame_to_rollout_idx = {}
if "rollout_frame_ids" in dataset.files:
    for case_idx, case in enumerate(rollout_case_names):
        frame_ids = [
            str(x).zfill(6) for x in dataset["rollout_frame_ids"][case_idx].tolist()
        ]
        case_frame_to_rollout_idx[case] = {fid: j for j, fid in enumerate(frame_ids)}
else:
    for case_idx, case in enumerate(rollout_case_names):
        seq = np.asarray(rollout_ground_truth[case_idx])
        frame_ids = sorted(set(case_to_frame_ids_from_pairs.get(case, [])))
        if len(frame_ids) >= seq.shape[0]:
            frame_ids = frame_ids[:seq.shape[0]]
        else:
            frame_ids = [str(j).zfill(6) for j in range(seq.shape[0])]
        case_frame_to_rollout_idx[case] = {fid: j for j, fid in enumerate(frame_ids)}


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    return np.clip(
        (positions - position_min[None, :]) / position_span[None, :], 0.0, 1.0
    ).astype(np.float32)


def normalize_positions_torch(positions: torch.Tensor) -> torch.Tensor:
    pmin = torch.tensor(position_min, dtype=torch.float32, device=positions.device).view(1, 1, 3)
    pspan = torch.tensor(position_span, dtype=torch.float32, device=positions.device).view(1, 1, 3)
    return torch.clamp((positions - pmin) / pspan, 0.0, 1.0)


def build_global_params(metadata: dict, phase_override: Optional[float] = None) -> np.ndarray:
    freestream = np.asarray(
        metadata.get("freestream", [0.0, 0.0, 0.0]), dtype=np.float32
    ).reshape(-1)
    if freestream.size < 3:
        freestream = np.pad(freestream, (0, 3 - freestream.size))

    values = {
        "angle_of_attack": float(metadata.get("aoa_deg", 0.0)) / 45.0,
        "freestream_magnitude": float(np.linalg.norm(freestream)) / 10.0,
        "freestream_x": float(freestream[0]) / 10.0,
        "freestream_y": float(freestream[1]) / 10.0,
        "freestream_z": float(freestream[2]) / 10.0,
        "phase": float(
            metadata.get("phase_t", 0.0) if phase_override is None else phase_override
        ),
    }
    return np.asarray([values[n] for n in global_param_names], dtype=np.float32)


def build_gno_block(in_channels: int, out_channels: int, radius: float):
    kwargs = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        coord_dim=3,
        radius=float(radius),
        transform_type="linear",
        reduction="mean",
        pos_embedding_type="transformer",
        pos_embedding_channels=12,
        channel_mlp_layers=[out_channels, out_channels, out_channels],
    )
    accepted = set(inspect.signature(GNOBlock.__init__).parameters)
    if "use_torch_scatter_reduce" in accepted:
        kwargs["use_torch_scatter_reduce"] = False
    if "use_open3d_neighbor_search" in accepted:
        kwargs["use_open3d_neighbor_search"] = False
    return GNOBlock(**{k: v for k, v in kwargs.items() if k in accepted})


def fourier_positional_encoding(coords: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    if int(num_frequencies) <= 0:
        return coords
    freqs = (
        2.0 ** torch.arange(int(num_frequencies), device=coords.device, dtype=coords.dtype)
    ).view(1, 1, -1)
    angles = coords.unsqueeze(-1) * freqs * math.pi
    return torch.cat(
        [coords, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1
    )


def build_latent_grid(resolution: int, device: torch.device) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, int(resolution), dtype=torch.float32, device=device)
    xx, yy, zz = torch.meshgrid(line, line, line, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)


class GINOSharedLatent(nn.Module):
    def __init__(
        self,
        in_channels: int,
        delta_channels: int,
        field_channels: int,
        global_channels: int,
        cfg_local: dict,
    ):
        super().__init__()
        hidden = int(cfg_local["hidden_channels"])
        self.latent_res = int(cfg_local["latent_res"])
        self.query_pe_freqs = int(cfg_local["query_pos_encoding_frequencies"])
        self.hidden = hidden

        self.lift = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.global_condition_mlp = nn.Sequential(
            nn.Linear(global_channels, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.encoder = build_gno_block(hidden, hidden, cfg_local["gno_radius"])

        modes = min(int(cfg_local["fno_modes"]), max(self.latent_res // 2, 1))
        self.fno = FNO(
            n_modes=(modes, modes, modes),
            in_channels=hidden,
            out_channels=hidden,
            hidden_channels=hidden,
            n_layers=int(cfg_local["fno_layers"]),
            positional_embedding=None,
        )

        query_dim = 3 + 2 * 3 * self.query_pe_freqs
        self.particle_pos_enc_proj = nn.Linear(query_dim, hidden)
        self.grid_pos_enc_proj = nn.Linear(query_dim, hidden)

        width = int(cfg_local["mlp_hidden"])
        decoder_layers = []
        for layer_id in range(max(int(cfg_local["mlp_layers"]), 1)):
            decoder_layers += [
                nn.Linear(hidden + query_dim if layer_id == 0 else width, width),
                nn.GELU(),
            ]
        decoder_layers.append(nn.Linear(width, field_channels))
        self.field_decoder = nn.Sequential(*decoder_layers)

        self.particle_skip_proj = nn.Linear(hidden, hidden)
        self.particle_query_proj = nn.Linear(hidden, hidden)
        self.grid_kv_proj = nn.Linear(hidden, 2 * hidden)
        self.attn_dropout = nn.Dropout(0.0)
        self.attn_norm = nn.LayerNorm(hidden)
        self.delta_fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, delta_channels)
        )

        self.log_delta_var = nn.Parameter(torch.zeros(()))
        self.log_field_var = nn.Parameter(torch.zeros(()))

    def apply_gno(self, block, source_coords, query_coords, source_features):
        return block(y=source_coords, x=query_coords, f_y=source_features)

    def encode_process(self, input_geom, latent_queries, x, global_params):
        base_latent = latent_queries[0]
        r = self.latent_res
        processed_grids = []
        flattened_latents = []
        particle_skips = []

        for b in range(x.shape[0]):
            h_raw = self.lift(x[b])
            latent = self.apply_gno(self.encoder, input_geom[b], base_latent, h_raw)
            if latent.ndim == 3:
                latent = latent.squeeze(0)

            grid = latent.reshape(r, r, r, -1).permute(3, 0, 1, 2).unsqueeze(0)
            condition = self.global_condition_mlp(global_params[b]).view(1, -1, 1, 1, 1)
            processed_grid = self.fno(grid + condition)
            processed_flat = (
                processed_grid.squeeze(0)
                .permute(1, 2, 3, 0)
                .reshape(-1, processed_grid.shape[1])
            )

            processed_grids.append(processed_grid)
            flattened_latents.append(processed_flat)
            particle_skips.append(self.particle_skip_proj(h_raw))

        return base_latent, processed_grids, flattened_latents, particle_skips

    def sample_grid(self, grid, queries):
        q = queries.clamp(0.0, 1.0)
        sample_coords = (q * 2.0 - 1.0).view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(grid, sample_coords, align_corners=True, mode="bilinear")
        return sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)

    def cross_attention(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, values)

    def delta_from_latents(
        self, base_latent, flattened_latents, particle_skips, input_geom
    ):
        grid_pos_enc = fourier_positional_encoding(
            base_latent.unsqueeze(0), self.query_pe_freqs
        ).squeeze(0)

        delta_outputs = []
        for b, (flat_latent, skip) in enumerate(zip(flattened_latents, particle_skips)):
            particle_pos_enc = fourier_positional_encoding(
                input_geom[b].unsqueeze(0), self.query_pe_freqs
            ).squeeze(0)

            q_base = skip + self.particle_pos_enc_proj(particle_pos_enc)
            q = self.particle_query_proj(q_base)

            kv_input = flat_latent + self.grid_pos_enc_proj(grid_pos_enc)
            kv = self.grid_kv_proj(kv_input)
            k = kv[..., : self.hidden]
            v = kv[..., self.hidden :]

            particle_latent = self.cross_attention(q, k, v)
            particle_latent = self.attn_norm(particle_latent + q)
            fused = self.delta_fusion(torch.cat([particle_latent, skip], dim=-1))
            delta_outputs.append(self.delta_head(fused).unsqueeze(0))

        return torch.cat(delta_outputs, dim=0)

    def forward(
        self, input_geom, latent_queries, output_queries, x, global_params
    ):
        base_latent, grids, flattened_latents, particle_skips = self.encode_process(
            input_geom, latent_queries, x, global_params
        )
        delta = self.delta_from_latents(
            base_latent, flattened_latents, particle_skips, input_geom
        )

        field_outputs = []
        for b, grid in enumerate(grids):
            q_field = output_queries[b].clamp(0.0, 1.0)
            sampled = self.sample_grid(grid, q_field)
            field_out = self.field_decoder(
                torch.cat(
                    [
                        sampled,
                        fourier_positional_encoding(
                            q_field.unsqueeze(0), self.query_pe_freqs
                        ).squeeze(0),
                    ],
                    dim=-1,
                )
            )
            field_outputs.append(field_out.unsqueeze(0))

        return delta, torch.cat(field_outputs, dim=0)


LATENT_GRID = build_latent_grid(model_config["latent_res"], DEVICE)

model = GINOSharedLatent(
    len(model_input_features),
    num_delta_channels,
    num_field_channels,
    len(global_param_names),
    model_config,
).to(DEVICE)

state_dict = dict(checkpoint["model_state_dict"])
state_dict.pop("_metadata", None)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys or unexpected_keys:
    raise RuntimeError(
        f"Checkpoint/model mismatch.\nMissing: {missing_keys}\nUnexpected: {unexpected_keys}"
    )

model.eval()
print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters.")


def build_inference_batch(
    pair_id: int,
    max_particles: int = MAX_MODEL_INPUT_PARTICLES,
    max_queries: int = MAX_MODEL_QUERY_POINTS,
) -> dict:
    pair_id = int(pair_id)
    start, end = int(pair_frame_ranges[pair_id][3]), int(pair_frame_ranges[pair_id][4])
    raw_inputs = all_input_features[start:end]

    metadata = pair_metadata_map[pair_id]
    case_name = str(metadata.get("case", "unknown"))

    n_total = raw_inputs.shape[0]
    n_min = int(case_min_particle_count.get(case_name, n_total))
    n_use = min(n_total, n_min, int(max_particles))
    particle_idx = np.arange(n_use, dtype=np.int64)

    x_raw = raw_inputs[particle_idx][:, model_input_indices]
    if x_raw.shape != (n_use, len(model_input_indices)):
        raise RuntimeError(
            f"Input feature extraction produced {x_raw.shape}; "
            f"expected {(n_use, len(model_input_indices))}."
        )

    x_norm = (x_raw - feature_mean[None, :]) / feature_std[None, :]
    x_norm = np.clip(np.nan_to_num(x_norm), -8.0, 8.0).astype(np.float32)

    position_raw = raw_inputs[particle_idx][:, position_indices]
    input_geom = normalize_positions(position_raw)

    valid_query_idx = np.flatnonzero(field_query_mask[pair_id])
    if valid_query_idx.size == 0:
        raise RuntimeError(f"pair_id={pair_id} has no valid field query points.")

    if (
        max_queries is None
        or int(max_queries) <= 0
        or valid_query_idx.size <= int(max_queries)
    ):
        query_idx = valid_query_idx
    else:
        chosen_positions = np.linspace(
            0, valid_query_idx.size - 1, int(max_queries), dtype=np.int64
        )
        query_idx = valid_query_idx[chosen_positions]

    query_xyz_raw = np.asarray(all_field_query_coords[pair_id, query_idx], dtype=np.float32)
    field_target_norm = np.asarray(
        normalized_field_targets[pair_id, query_idx], dtype=np.float32
    )

    state_phys = np.asarray(raw_inputs[particle_idx][:, particle_state_indices], dtype=np.float32)
    velocity_phys = np.asarray(raw_inputs[particle_idx][:, velocity_indices], dtype=np.float32)
    gradient_phys = np.asarray(
        raw_inputs[particle_idx][:, velocity_gradient_indices], dtype=np.float32
    )
    delta_target_norm = np.asarray(
        normalized_delta_targets[start:end][particle_idx][:, :num_delta_channels], dtype=np.float32
    )
    delta_target_phys = np.asarray(
        physical_delta_targets[start:end][particle_idx][:, :num_delta_channels], dtype=np.float32
    )

    return {
        "input_geom": torch.from_numpy(input_geom).unsqueeze(0).to(DEVICE),
        "x": torch.from_numpy(x_norm).unsqueeze(0).to(DEVICE),
        "output_queries": torch.from_numpy(normalize_positions(query_xyz_raw))
        .unsqueeze(0)
        .to(DEVICE),
        "global_params": torch.from_numpy(build_global_params(metadata))
        .unsqueeze(0)
        .to(DEVICE),
        "state_phys": state_phys,
        "velocity_phys": velocity_phys,
        "gradU_phys": gradient_phys,
        "delta_target_norm": delta_target_norm,
        "delta_target_phys": delta_target_phys,
        "field_target_norm": field_target_norm,
        "query_xyz_raw": query_xyz_raw,
        "input_xyz_raw": position_raw,
        "dt": float(metadata.get("dt", 0.0034)),
        "pair_id": torch.tensor([pair_id], dtype=torch.long, device=DEVICE),
        "case": case_name,
    }


def run_model(batch: dict):
    return model(
        batch["input_geom"],
        LATENT_GRID,
        batch["output_queries"],
        batch["x"],
        batch["global_params"],
    )


def denormalize_delta(prediction: torch.Tensor) -> torch.Tensor:
    return prediction * delta_std_t + delta_mean_t


def denormalize_field(prediction: torch.Tensor) -> torch.Tensor:
    return prediction * field_std_t + field_mean_t


def compute_euler_baseline(batch: dict):
    state_t = torch.from_numpy(batch["state_phys"]).to(DEVICE, dtype=torch.float32)
    velocity_t = torch.from_numpy(batch["velocity_phys"]).to(DEVICE, dtype=torch.float32)
    grad_u_t = (
        torch.from_numpy(batch["gradU_phys"]).to(DEVICE, dtype=torch.float32).reshape(-1, 3, 3)
    )
    dt = float(batch.get("dt", 0.0034))
    dx_euler = dt * velocity_t
    dgamma_euler = dt * torch.einsum("nij,nj->ni", grad_u_t, state_t[:, 3:6])
    return state_t, dx_euler, dgamma_euler


def integrate_state_with_residual(batch: dict, residual_norm_prediction: torch.Tensor):
    residual_phys = denormalize_delta(residual_norm_prediction)
    state_t, dx_euler, dgamma_euler = compute_euler_baseline(batch)

    state_next_pred = state_t.clone()
    state_next_pred[:, :3] += dx_euler + residual_phys[:, :3]
    state_next_pred[:, 3:6] += dgamma_euler + residual_phys[:, 3:6]
    state_next_pred[:, 6] += residual_phys[:, 6]

    raw_delta = torch.from_numpy(batch["delta_target_phys"][:, :7]).to(
        DEVICE, dtype=torch.float32
    )
    state_next_true = state_t + raw_delta
    return state_next_pred, state_next_true, residual_phys


def compute_one_step_errors() -> pd.DataFrame:
    records = []
    with torch.inference_mode():
        for pid in test_pair_ids:
            pid = int(pid)
            batch = build_inference_batch(pid)
            delta_pred, _ = run_model(batch)
            state_pred, state_true, _ = integrate_state_with_residual(batch, delta_pred[0])

            pred_np = state_pred.cpu().numpy()
            true_np = state_true.cpu().numpy()

            pos_diff = pred_np[:, :3] - true_np[:, :3]
            circ_diff = pred_np[:, 3:6] - true_np[:, 3:6]
            sigma_diff = pred_np[:, 6] - true_np[:, 6]

            records.append(
                {
                    "pair_id": pid,
                    "phase": float(pair_metadata_map[pid].get("phase_t", 0.0)),
                    "aoa": int(round(float(pair_metadata_map[pid].get("aoa_deg", 0.0)))),
                    "position_rmse": float(np.sqrt(np.mean(np.sum(pos_diff ** 2, axis=1)))),
                    "circulation_rmse": float(
                        np.sqrt(np.mean(np.sum(circ_diff ** 2, axis=1)))
                    ),
                    "sigma_rmse": float(np.sqrt(np.mean(sigma_diff ** 2))),
                }
            )
    return pd.DataFrame(records)


one_step_error_table = compute_one_step_errors()

one_step_metric_specs = [
    ("position_rmse", "Position RMSE (m)"),
    ("circulation_rmse", "Circulation RMSE"),
    ("sigma_rmse", "Sigma RMSE"),
]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, constrained_layout=True)

for axis, (metric_key, metric_label) in zip(axes, one_step_metric_specs):
    for aoa, marker in zip(ONE_STEP_ANGLES_OF_ATTACK, ("o", "s")):
        df = one_step_error_table[
            (one_step_error_table["aoa"] == aoa)
            & (one_step_error_table["phase"] >= MIN_PHASE_TO_PLOT)
        ].sort_values("phase")
        if df.empty:
            continue
        axis.plot(
            df["phase"].to_numpy(),
            df[metric_key].to_numpy(),
            marker=marker,
            linewidth=1.1,
            markersize=2.7,
            markerfacecolor="white",
            label=f"AoA = {aoa}°",
        )
    axis.set_xlabel("Phase", fontsize=8)
    axis.set_ylabel(metric_label, fontsize=8)
    axis.grid(True, linewidth=0.45, alpha=0.3, linestyle="--")
    axis.tick_params(labelsize=7)

axes[0].legend(frameon=False, fontsize=7, loc="best")
fig.suptitle("One-step prediction error", fontsize=10)

one_step_figure_path = PLOT_DIR / "fig_A_one_step_RMSE.pdf"
fig.savefig(one_step_figure_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved Figure A: {one_step_figure_path}")


def lookup_rollout_frame_index(pair_id: int) -> Tuple[str, int]:
    pair_id = int(pair_id)
    metadata = pair_metadata_map[pair_id]
    case_name = str(metadata.get("case", ""))
    frame_id = str(metadata.get("frame_t", metadata.get("frame_id", ""))).zfill(6)

    case_idx = case_to_rollout_idx.get(case_name)
    if case_idx is None:
        raise RuntimeError(f"No rollout_true_states entry for case {case_name!r}.")

    frame_map = case_frame_to_rollout_idx.get(case_name, {})
    start_idx = frame_map.get(frame_id)
    if start_idx is None:
        raise RuntimeError(
            f"Could not map case={case_name!r}, frame={frame_id} to rollout_true_states."
        )
    return case_name, int(start_idx)


def get_rollout_ground_truth(pair_id: int, steps: int):
    case_name, start_idx = lookup_rollout_frame_index(pair_id)
    case_idx = case_to_rollout_idx[case_name]
    states = np.asarray(rollout_ground_truth[case_idx], dtype=np.float32)

    if states.ndim != 3:
        raise RuntimeError(
            f"rollout_true_states[{case_idx}] has shape {states.shape}; expected (T,N,7)."
        )

    max_available = max(int(states.shape[0]) - start_idx - 1, 0)
    actual_steps = min(int(steps), max_available)
    if actual_steps <= 0:
        return None

    targets = states[
        start_idx + 1 : start_idx + 1 + actual_steps, :MAX_MODEL_INPUT_PARTICLES, :7
    ]

    phases = None
    if len(rollout_phase_values) > case_idx:
        phases_all = np.asarray(rollout_phase_values[case_idx], dtype=np.float32)
        phases = phases_all[start_idx + 1 : start_idx + 1 + actual_steps]

    return targets, phases


def select_rollout_starting_pairs(
    aoa: int, desired_phases: Sequence[float], max_steps: int
):
    candidates = []
    for pid in test_pair_ids:
        pid = int(pid)
        metadata = pair_metadata_map[pid]
        pid_aoa = int(round(float(metadata.get("aoa_deg", 0.0))))
        if pid_aoa != int(aoa):
            continue
        try:
            rollout_data = get_rollout_ground_truth(pid, max_steps)
        except RuntimeError:
            continue
        if rollout_data is None or rollout_data[0].shape[0] < max_steps:
            continue
        candidates.append((pid, float(metadata.get("phase_t", 0.0))))

    if not candidates:
        raise RuntimeError(
            f"No eligible rollout starts for AoA={aoa}° and horizon={max_steps}."
        )

    selected, used = [], set()
    for requested_phase in desired_phases:
        remaining = [c for c in candidates if c[0] not in used]
        if not remaining:
            break
        pid, actual_phase = min(
            remaining, key=lambda c: (abs(c[1] - float(requested_phase)), c[0])
        )
        selected.append((pid, actual_phase, float(requested_phase)))
        used.add(pid)
    return selected


def select_pairs_nearest_phase(aoa: int, desired_phases: Sequence[float]):
    candidates = []
    for pid in test_pair_ids:
        pid = int(pid)
        metadata = pair_metadata_map[pid]
        pid_aoa = int(round(float(metadata.get("aoa_deg", 0.0))))
        if pid_aoa != int(aoa):
            continue
        candidates.append((pid, float(metadata.get("phase_t", 0.0))))

    if not candidates:
        raise RuntimeError(f"No test pairs found for AoA={aoa}°.")

    selected, used = [], set()
    for requested_phase in desired_phases:
        remaining = [c for c in candidates if c[0] not in used]
        if not remaining:
            raise RuntimeError(
                f"Could not select distinct pairs for all requested phases at AoA={aoa}°."
            )
        pid, actual_phase = min(
            remaining, key=lambda c: (abs(c[1] - float(requested_phase)), c[0])
        )
        selected.append((pid, actual_phase, float(requested_phase)))
        used.add(pid)

    for pid, actual_phase, requested_phase in selected:
        print(
            f"AoA={aoa}°: requested phase={requested_phase:.3f} "
            f"-> pair_id={pid}, actual phase={actual_phase:.6f}"
        )
    return selected


def rebuild_model_input_from_state(template_x, state_phys, velocity_phys, grad_u_phys):
    B, N, _ = template_x.shape
    mean_full = torch.tensor(
        feature_mean, dtype=template_x.dtype, device=template_x.device
    ).view(1, 1, -1)
    std_full = torch.tensor(
        feature_std, dtype=template_x.dtype, device=template_x.device
    ).view(1, 1, -1)

    old_active_phys = template_x * std_full + mean_full
    full_raw = torch.zeros(
        (B, N, len(all_feature_names)), dtype=template_x.dtype, device=template_x.device
    )

    for local_i, global_i in enumerate(model_input_indices):
        full_raw[:, :, global_i] = old_active_phys[:, :, local_i]

    full_raw[:, :, particle_state_indices] = state_phys
    full_raw[:, :, velocity_indices] = velocity_phys
    full_raw[:, :, velocity_gradient_indices] = grad_u_phys

    active = full_raw[:, :, model_input_indices]
    normalized = (active - mean_full) / std_full
    return torch.clamp(torch.nan_to_num(normalized), -8.0, 8.0)


def update_phase_in_global_params(global_params, phase_value):
    updated = global_params.clone()
    if phase_value is not None and "phase" in global_param_names:
        phase_idx = global_param_names.index("phase")
        updated[:, phase_idx] = float(phase_value)
    return updated


@torch.inference_mode()
def run_autoregressive_rollout(start_pair_id: int, max_steps: int):
    pair_id = int(start_pair_id)
    rollout_data = get_rollout_ground_truth(pair_id, max_steps)
    if rollout_data is None:
        return None

    target_states_np, target_phases_np = rollout_data
    if target_states_np.shape[0] < max_steps:
        return None

    initial_batch = build_inference_batch(pair_id)

    state_phys = (
        torch.from_numpy(initial_batch["state_phys"]).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    )
    velocity_phys = (
        torch.from_numpy(initial_batch["velocity_phys"])
        .unsqueeze(0)
        .to(DEVICE, dtype=torch.float32)
    )
    grad_u_phys = (
        torch.from_numpy(initial_batch["gradU_phys"])
        .unsqueeze(0)
        .to(DEVICE, dtype=torch.float32)
        .reshape(1, -1, 3, 3)
    )

    geom = initial_batch["input_geom"].clone()
    x = initial_batch["x"].clone()
    template_x = initial_batch["x"].clone()
    global_params = initial_batch["global_params"].clone()
    dt = float(initial_batch["dt"])

    pos_min_t = torch.tensor(position_min, dtype=torch.float32, device=DEVICE).view(1, 1, 3)
    pos_max_t = pos_min_t + torch.tensor(
        position_span, dtype=torch.float32, device=DEVICE
    ).view(1, 1, 3)

    step_records = []

    for step_idx in range(max_steps):
        delta_pred, field_particle_pred = model(
            geom, LATENT_GRID, geom, x, global_params
        )
        delta_phys = denormalize_delta(delta_pred)[:, :, :7]
        field_phys = denormalize_field(field_particle_pred)

        velocity_current = field_phys[..., :3]
        grad_current = field_phys[..., 3:12].reshape(1, -1, 3, 3)
        gamma_current = state_phys[:, :, 3:6]

        dx_euler = dt * velocity_current
        dgamma_euler = dt * torch.einsum("bnij,bnj->bni", grad_current, gamma_current)

        next_state = state_phys.clone()
        next_state[:, :, :3] = state_phys[:, :, :3] + dx_euler + delta_phys[:, :, :3]
        next_state[:, :, 3:6] = (
            state_phys[:, :, 3:6] + dgamma_euler + delta_phys[:, :, 3:6]
        )
        next_state[:, :, 6] = state_phys[:, :, 6] + delta_phys[:, :, 6]

        target_step = (
            torch.from_numpy(target_states_np[step_idx, : next_state.shape[1], :7])
            .unsqueeze(0)
            .to(DEVICE, dtype=torch.float32)
        )

        err = next_state - target_step
        pos_rmse = torch.sqrt(torch.mean(torch.sum(err[:, :, :3] ** 2, dim=-1)))
        circ_rmse = torch.sqrt(torch.mean(torch.sum(err[:, :, 3:6] ** 2, dim=-1)))
        sigma_rmse = torch.sqrt(torch.mean(err[:, :, 6] ** 2))

        step_records.append(
            {
                "step": step_idx + 1,
                "position_rmse": float(pos_rmse.item()),
                "circulation_rmse": float(circ_rmse.item()),
                "sigma_rmse": float(sigma_rmse.item()),
                "phase": (
                    float(target_phases_np[step_idx])
                    if target_phases_np is not None
                    else np.nan
                ),
            }
        )

        state_phys = next_state
        geom = normalize_positions_torch(state_phys[:, :, :3])

        phase_value = (
            None if target_phases_np is None else float(target_phases_np[step_idx])
        )
        x = rebuild_model_input_from_state(
            template_x, state_phys, velocity_current, grad_current.reshape(1, -1, 9)
        )
        global_params = update_phase_in_global_params(global_params, phase_value)

    return pd.DataFrame(step_records)


rollout_results = []
for aoa in ROLLOUT_ANGLES_OF_ATTACK:
    starts = select_rollout_starting_pairs(aoa, ROLLOUT_START_PHASES, ROLLOUT_HORIZON)
    print(f"\nAoA={aoa}° rollout starts:")
    for pid, actual_phase, requested_phase in starts:
        print(
            f"   requested phase={requested_phase:.2f} -> pair={pid}, "
            f"actual phase={actual_phase:.6f}"
        )
        records = run_autoregressive_rollout(pid, ROLLOUT_HORIZON)
        if records is None:
            print(f"   WARNING: rollout skipped for pair {pid}.")
            continue
        rollout_results.append(
            {
                "aoa": aoa,
                "start_pair_id": pid,
                "start_phase": actual_phase,
                "records": records,
            }
        )

if not rollout_results:
    raise RuntimeError("No autoregressive rollout was successfully evaluated.")

rollout_error_table = pd.concat(
    [
        r["records"].assign(
            aoa=r["aoa"], start_pair_id=r["start_pair_id"], start_phase=r["start_phase"]
        )
        for r in rollout_results
    ],
    ignore_index=True,
)


def aggregate_rollout_stats(df: pd.DataFrame, metric: str):
    return df.groupby(["aoa", "step"], as_index=False)[metric].agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75),
    )


rollout_metric_specs = [
    ("position_rmse", "Position RMSE (m)"),
    ("circulation_rmse", "Circulation RMSE"),
    ("sigma_rmse", "Sigma RMSE"),
]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, constrained_layout=True)

for axis, (metric_key, metric_label) in zip(axes, rollout_metric_specs):
    for aoa, marker in zip(ROLLOUT_ANGLES_OF_ATTACK, ("o", "s")):
        stats = aggregate_rollout_stats(
            rollout_error_table[rollout_error_table["aoa"] == aoa], metric_key
        )
        if stats.empty:
            continue
        axis.plot(
            stats["step"].to_numpy(),
            stats["median"].to_numpy(),
            marker=marker,
            linewidth=1.1,
            markersize=2.5,
            markerfacecolor="white",
            markevery=max(1, len(stats) // 10),
            label=f"AoA = {aoa}°",
        )
    axis.set_xlabel("Rollout step", fontsize=8)
    axis.set_ylabel(metric_label, fontsize=8)
    axis.grid(True, linewidth=0.45, alpha=0.3, linestyle="--")
    axis.tick_params(labelsize=7)

axes[0].legend(frameon=False, fontsize=7, loc="best")
fig.suptitle("Autoregressive rollout error", fontsize=10)

rollout_figure_path = PLOT_DIR / "fig_B_autoregressive_rollout_RMSE.pdf"
fig.savefig(rollout_figure_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved Figure B: {rollout_figure_path}")

rollout_error_table[
    [
        "aoa",
        "start_pair_id",
        "start_phase",
        "step",
        "phase",
        "position_rmse",
        "circulation_rmse",
        "sigma_rmse",
    ]
].to_csv(PLOT_DIR / "fig_B_rollout_RMSE_data.csv", index=False)

summary_rows = []
for aoa in ROLLOUT_ANGLES_OF_ATTACK:
    df_aoa = rollout_error_table[rollout_error_table["aoa"] == aoa]
    for metric_key, metric_label in rollout_metric_specs:
        final_values = df_aoa[df_aoa["step"] == ROLLOUT_HORIZON][metric_key].to_numpy()
        time_avg_values = df_aoa.groupby("start_pair_id")[metric_key].mean().to_numpy()
        summary_rows.append(
            {
                "AoA": aoa,
                "metric": metric_label,
                "final_step_median": float(np.median(final_values))
                if final_values.size
                else np.nan,
                "final_step_mean": float(np.mean(final_values))
                if final_values.size
                else np.nan,
                "time_averaged_rmse_median": float(np.median(time_avg_values))
                if time_avg_values.size
                else np.nan,
                "time_averaged_rmse_mean": float(np.mean(time_avg_values))
                if time_avg_values.size
                else np.nan,
            }
        )

rollout_summary = pd.DataFrame(summary_rows)
rollout_summary.to_csv(PLOT_DIR / "fig_B_rollout_summary.csv", index=False)
print("\nAutoregressive rollout summary:")
print(rollout_summary.to_string(index=False))


# NOTE: `_load_native_field(pair_id)` must be defined or imported externally.
# It is expected to return: (hdf5_path_str, node_positions_Nx3, node_velocities_Nx3, freestream_speed_scalar)


def extract_native_slice_on_common_grid(
    pair_id: int, y_slice: float, resolution: int = 256, y_tolerance: float = 0.01
):
    path, nodes, velocity, freestream = _load_native_field(pair_id)

    y_coords = nodes[:, 1]
    y_distance = np.abs(y_coords - float(y_slice))
    mask = y_distance <= float(y_tolerance)

    if int(mask.sum()) < 100:
        nearest_distance = float(y_distance.min())
        mask = y_distance <= max(float(y_tolerance), 2.0 * nearest_distance)

    if int(mask.sum()) < 100:
        raise RuntimeError(
            f"{path}: only {int(mask.sum())} native points near y={y_slice:.8f}."
        )

    x = nodes[mask, 0]
    z = nodes[mask, 2]
    u_magnitude_nd = np.linalg.norm(velocity[mask], axis=1) / freestream

    x_lo, x_hi = float(x.min()), float(x.max())
    z_lo, z_hi = float(z.min()), float(z.max())
    if not (x_hi > x_lo and z_hi > z_lo):
        raise RuntimeError(f"{path}: degenerate x-z native slice.")

    x_vals = np.linspace(x_lo, x_hi, int(resolution), dtype=np.float32)
    z_vals = np.linspace(z_lo, z_hi, int(resolution), dtype=np.float32)
    XX, ZZ = np.meshgrid(x_vals, z_vals, indexing="xy")

    true_grid = griddata(
        (x, z), u_magnitude_nd, (XX, ZZ), method="linear", fill_value=np.nan
    )
    nan_mask = np.isnan(true_grid)
    if nan_mask.any():
        nearest_grid = griddata((x, z), u_magnitude_nd, (XX, ZZ), method="nearest")
        true_grid[nan_mask] = nearest_grid[nan_mask]

    metadata = pair_metadata_map[int(pair_id)]
    print(f"   HDF5 true field: {path}")
    print(f"   Exact frame: {int(str(metadata.get('frame_t', '0')))}")
    print(f"   Native points near y={y_slice:.8f}: {int(mask.sum())}")
    print(f"   Common grid: x=[{x_lo:.6f}, {x_hi:.6f}], z=[{z_lo:.6f}, {z_hi:.6f}]")

    return x_vals, z_vals, true_grid.astype(np.float32), freestream


@torch.inference_mode()
def predict_field_on_slice(
    pair_id: int,
    x_vals: np.ndarray,
    z_vals: np.ndarray,
    y_slice: float,
    query_chunk_size: int = 32768,
):
    batch = build_inference_batch(pair_id)
    x_vals = np.asarray(x_vals, dtype=np.float32)
    z_vals = np.asarray(z_vals, dtype=np.float32)

    XX, ZZ = np.meshgrid(x_vals, z_vals, indexing="xy")
    YY = np.full_like(XX, float(y_slice), dtype=np.float32)
    queries = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1).astype(np.float32)

    _, grids, _, _ = model.encode_process(
        batch["input_geom"], LATENT_GRID, batch["x"], batch["global_params"]
    )
    grid = grids[0]
    magnitude_chunks = []
    chunk_size = max(int(query_chunk_size), 1)

    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        q_chunk_np = queries[start:stop]
        q_chunk = torch.from_numpy(normalize_positions(q_chunk_np)).to(
            DEVICE, dtype=torch.float32
        )

        sampled = model.sample_grid(grid, q_chunk)
        field_chunk = model.field_decoder(
            torch.cat(
                [
                    sampled,
                    fourier_positional_encoding(
                        q_chunk.unsqueeze(0), model.query_pe_freqs
                    ).squeeze(0),
                ],
                dim=-1,
            )
        )
        velocity_chunk = field_chunk[:, :3] * field_std_t[:3] + field_mean_t[:3]
        magnitude_chunk = torch.linalg.norm(velocity_chunk, dim=1).cpu().numpy()
        magnitude_chunks.append(magnitude_chunk.astype(np.float32))

    magnitude = np.concatenate(magnitude_chunks, axis=0).reshape(len(z_vals), len(x_vals))

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return XX, ZZ, magnitude


def plot_field_comparison_figure(
    aoa: int,
    selected_phases: Sequence[float],
    profile_x_values: Sequence[float],
    z_clip_max: float,
    field_resolution: int = 256,
):
    if len(selected_phases) != len(profile_x_values):
        raise ValueError("selected_phases and profile_x_values must have equal length.")

    selections = select_pairs_nearest_phase(aoa, selected_phases)
    if not selections:
        raise RuntimeError(f"No field-slice selections for AoA={aoa}°.")

    phase_snapshots = []
    for requested_phase, profile_x, selected in zip(
        selected_phases, profile_x_values, selections
    ):
        pid, actual_phase, _ = selected
        x_grid, z_grid, true_magnitude, freestream = extract_native_slice_on_common_grid(
            pid, y_slice=SLICE_Y, resolution=field_resolution
        )
        phase_snapshots.append(
            {
                "pid": pid,
                "requested_phase": float(requested_phase),
                "actual_phase": float(actual_phase),
                "profile_x": float(profile_x),
                "x_grid_native": x_grid,
                "z_grid_native": z_grid,
                "true_native": true_magnitude,
                "freestream": float(freestream),
            }
        )

    common_x_min = max(float(s["x_grid_native"].min()) for s in phase_snapshots)
    common_x_max = min(float(s["x_grid_native"].max()) for s in phase_snapshots)
    common_z_min = max(float(s["z_grid_native"].min()) for s in phase_snapshots)
    common_z_max = min(float(s["z_grid_native"].max()) for s in phase_snapshots)

    if not (common_x_max > common_x_min and common_z_max > common_z_min):
        raise RuntimeError(f"AoA={aoa}° snapshots have no common plotting domain.")

    x_common = np.linspace(
        common_x_min, common_x_max, int(field_resolution), dtype=np.float32
    )
    z_common = np.linspace(
        common_z_min, common_z_max, int(field_resolution), dtype=np.float32
    )
    XX_common, ZZ_common = np.meshgrid(x_common, z_common, indexing="xy")

    for snapshot in phase_snapshots:
        native_xx, native_zz = np.meshgrid(
            snapshot["x_grid_native"], snapshot["z_grid_native"], indexing="xy"
        )
        true_common = griddata(
            (native_xx.ravel(), native_zz.ravel()),
            snapshot["true_native"].ravel(),
            (XX_common, ZZ_common),
            method="linear",
            fill_value=np.nan,
        )
        nan_mask = np.isnan(true_common)
        if nan_mask.any():
            nearest_common = griddata(
                (native_xx.ravel(), native_zz.ravel()),
                snapshot["true_native"].ravel(),
                (XX_common, ZZ_common),
                method="nearest",
            )
            true_common[nan_mask] = nearest_common[nan_mask]

        _, _, pred_magnitude = predict_field_on_slice(
            snapshot["pid"], x_vals=x_common, z_vals=z_common, y_slice=SLICE_Y
        )
        snapshot["XX"] = XX_common
        snapshot["ZZ"] = ZZ_common
        snapshot["true"] = true_common.astype(np.float32)
        snapshot["pred"] = (pred_magnitude / snapshot["freestream"]).astype(np.float32)

    vmax = max(
        max(float(np.nanmax(s["true"])) for s in phase_snapshots),
        max(float(np.nanmax(s["pred"])) for s in phase_snapshots),
    )
    vmax = max(1.0e-8, 1.05 * vmax)

    ncols = len(phase_snapshots)
    fig, axes = plt.subplots(3, ncols, figsize=(7.4, 5.4), constrained_layout=True)
    axes = np.asarray(axes).reshape(3, ncols)

    first_image = None
    for col, snapshot in enumerate(phase_snapshots):
        axis = axes[0, col]
        first_image = axis.pcolormesh(
            snapshot["XX"],
            snapshot["ZZ"],
            snapshot["true"],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        axis.set_xlim(common_x_min, common_x_max)
        axis.set_ylim(common_z_min, common_z_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"phase={snapshot['actual_phase']:.2f}", fontsize=9, pad=4)

        axis = axes[1, col]
        axis.pcolormesh(
            snapshot["XX"],
            snapshot["ZZ"],
            snapshot["pred"],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        axis.set_xlim(common_x_min, common_x_max)
        axis.set_ylim(common_z_min, common_z_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])

        axis = axes[2, col]
        profile_x = snapshot["profile_x"]
        column_idx = int(np.argmin(np.abs(x_common - profile_x)))
        z_profile_mask = z_common <= float(z_clip_max)

        axis.plot(
            snapshot["true"][z_profile_mask, column_idx],
            z_common[z_profile_mask],
            linewidth=1.2,
            label="True",
        )
        axis.plot(
            snapshot["pred"][z_profile_mask, column_idx],
            z_common[z_profile_mask],
            linestyle="--",
            linewidth=1.2,
            label="Predicted",
        )

        axis.set_xlabel("$|u|/U_\\infty$", fontsize=8)
        axis.set_ylabel("z", fontsize=8)
        axis.set_ylim(common_z_min, min(common_z_max, float(z_clip_max)))
        axis.set_title(f"x={profile_x:.2f}", fontsize=9, pad=4)
        axis.grid(True, linewidth=0.4, alpha=0.25, linestyle="--")
        axis.tick_params(labelsize=7)

        if col == 0:
            axis.legend(frameon=False, fontsize=7, loc="best")

    axes[0, 0].set_ylabel("True", fontsize=8)
    axes[1, 0].set_ylabel("Predicted", fontsize=8)
    axes[2, 0].set_ylabel("Velocity profile\nz", fontsize=8)

    colorbar = fig.colorbar(
        first_image, ax=axes[:2, :].ravel().tolist(), shrink=0.88, pad=0.02
    )
    colorbar.set_label("$|u|/U_\\infty$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"AoA = {aoa}°", fontsize=11)

    output = PLOT_DIR / f"fig_field_slice_AoA{aoa}.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved field figure: {output}")


print("\nGenerating AoA = 21° native-HDF5 field comparison ...")
plot_field_comparison_figure(
    21, FIELD_FIGURE_PHASES, FIELD_FIGURE_PROFILE_STATIONS, PROFILE_Z_CLIP_AOA_21, 256
)

print("\nGenerating AoA = 32° native-HDF5 field comparison ...")
plot_field_comparison_figure(
    32, FIELD_FIGURE_PHASES, FIELD_FIGURE_PROFILE_STATIONS, PROFILE_Z_CLIP_AOA_32, 256
)

print("\n" + "=" * 72)
print("GINO evaluation complete.")
print(f"Figure A: {PLOT_DIR / 'fig_A_one_step_RMSE.pdf'}")
print(f"Figure B: {PLOT_DIR / 'fig_B_autoregressive_rollout_RMSE.pdf'}")
print(f"Field 21: {PLOT_DIR / 'fig_field_slice_AoA21.pdf'}")
print(f"Field 32: {PLOT_DIR / 'fig_field_slice_AoA32.pdf'}")
print(f"Rollout summary: {PLOT_DIR / 'fig_B_rollout_summary.csv'}")
print("=" * 72)