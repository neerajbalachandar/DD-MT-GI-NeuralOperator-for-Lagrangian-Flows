"""Simplified manual preprocessing (single file).

Task-1 focus in this version:
- Build particle EVOLUTION dataset using consecutive frame pairs.
- Learn x_t -> delta_state_t where delta_state_t = state_{t+1} - state_t.
- Save rollout-ready per-case sequences for autoregressive validation.

Task-2 (field/FNO) code path is kept optional and disabled by default.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ==============================================================================
# USER SETTINGS
# ==============================================================================
RAW_ROOT = Path("/media/dysco/New Volume/Neeraj/neuralop/data/raw data")
DATASET_IDS = ["1", "2", "7", "8", "9"]

INPUT_H5_PATTERN = "input/wing-example_pfield.*.h5"
OUTPUT_H5_PATTERN = "output/wing-example_fdom.*.h5"
VTK_PATTERN = "vtk/wing-example_Wing_vlm.*.vtk"

OUT_ROOT = Path(__file__).resolve().parent / "output"
MERGED_ROOT = OUT_ROOT / "merged_frames"

# Task-2 path is optional and currently off (user requested Task-1 focus).
RUN_TASK2_FIELD = False
GRID_RESOLUTION = (32, 32, 32)
OUTPUT_MODE = "UW"
FIELD_INPUT_CHANNELS = [
    "Gamma_x",
    "Gamma_y",
    "Gamma_z",
    "sigma",
    "density",
    "X",
    "Y",
    "Z",
]

# Task-1 evolution setup
RANDOM_SEED = 42

# IMPORTANT: fill these with real metadata.
# This removes fake placeholders and prevents silent leakage/assumptions.
CASE_METADATA = {
    "1": {"aoa_deg": None, "freestream": None, "dt": None},
    "2": {"aoa_deg": None, "freestream": None, "dt": None},
    "7": {"aoa_deg": None, "freestream": None, "dt": None},
    "8": {"aoa_deg": None, "freestream": None, "dt": None},
    "9": {"aoa_deg": None, "freestream": None, "dt": None},
}

# Split by CASE (or AoA groups) to avoid temporal leakage.
# Example desired pattern:
#   TRAIN_CASES = ["1", "2", "7"]
#   VAL_CASES   = ["8"]
#   TEST_CASES  = ["9"]
TRAIN_CASES = ["1", "2", "7"]
VAL_CASES = ["8"]
TEST_CASES = ["9"]

# Feature/state definitions for Task-1
STATE_NAMES = ["x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma"]
TARGET_DELTA_NAMES = [
    "dx",
    "dy",
    "dz",
    "dGamma_x",
    "dGamma_y",
    "dGamma_z",
    "dsigma",
]

PARTICLE_INPUT_FEATURES = [
    "x",
    "y",
    "z",
    "Gamma_x",
    "Gamma_y",
    "Gamma_z",
    "sigma",
]


# ==============================================================================
# Basic helpers
# ==============================================================================
FRAME_RE = re.compile(r"(\d+)(?!.*\d)")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def frame_id(path: Path) -> str:
    m = FRAME_RE.search(path.stem)
    return m.group(1).zfill(6) if m else path.stem


def as_xyz(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.astype(np.float64)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T.astype(np.float64)
    raise ValueError(f"Cannot parse xyz from shape={a.shape}")


def as_vec_field(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 4 and a.shape[-1] == 3:
        return a.astype(np.float64)
    if a.ndim == 4 and a.shape[0] == 3:
        return np.moveaxis(a, 0, -1).astype(np.float64)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.astype(np.float64)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T.astype(np.float64)
    raise ValueError(f"Cannot parse vector field from shape={a.shape}")


def read_h5_selected(path: Path, key_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for out_key, in_key in key_map.items():
            if in_key not in f:
                raise KeyError(f"{path.name}: missing key {in_key}")
            out[out_key] = np.asarray(f[in_key])
    return out


def _fit_scalar(arr: np.ndarray, n: int, default: float) -> np.ndarray:
    x = np.asarray(arr).reshape(-1)
    if x.size == 0:
        return np.full(n, default, dtype=np.float64)
    if x.size < n:
        out = np.full(n, default, dtype=np.float64)
        out[: x.size] = x
        return out
    return x[:n].astype(np.float64)


def _rows_from_pair_ids(pair_ranges: List[Tuple], pair_ids: np.ndarray) -> np.ndarray:
    chunks = []
    for i in pair_ids:
        _, _, _, s, e, _ = pair_ranges[int(i)]
        chunks.append(np.arange(int(s), int(e), dtype=np.int64))
    return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.int64)


def _validate_case_split(all_cases: List[str]) -> None:
    all_set = set(all_cases)
    tr, va, te = set(TRAIN_CASES), set(VAL_CASES), set(TEST_CASES)
    if tr & va or tr & te or va & te:
        raise ValueError("TRAIN_CASES / VAL_CASES / TEST_CASES must be disjoint")
    unknown = (tr | va | te) - all_set
    if unknown:
        raise ValueError(f"Split includes unknown cases: {sorted(unknown)}")
    uncovered = all_set - (tr | va | te)
    if uncovered:
        raise ValueError(f"Some cases are not assigned to any split: {sorted(uncovered)}")


def _case_meta(case: str) -> Dict[str, object]:
    """Return case metadata with safe fallbacks.

    Task-1 can run without external metadata when metadata channels are not used
    as input features.
    """
    meta = CASE_METADATA.get(case, {}) or {}

    aoa = meta.get("aoa_deg", 0.0)
    fs = meta.get("freestream", [0.0, 0.0, 0.0])
    dt = meta.get("dt", 1.0)

    fs_arr = np.asarray(fs, dtype=np.float64).reshape(-1)
    if fs_arr.shape[0] != 3:
        fs_arr = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    return {
        "aoa_deg": float(aoa),
        "freestream": fs_arr,
        "dt": float(dt),
    }


# ==============================================================================
# Stage-1 merge
# ==============================================================================
INPUT_KEYS = {
    "particle_xyz": "X",
    "Gamma_vec": "Gamma",
    "velocity": "velocity",
    "velocity_gradient_x": "velocity_gradient_x",
    "velocity_gradient_y": "velocity_gradient_y",
    "velocity_gradient_z": "velocity_gradient_z",
    "sigma": "sigma",
    "circulation": "circulation",
    "vol": "vol",
    "static": "static",
}
OUTPUT_KEYS = {"points": "nodes", "U": "U", "W": "W"}


def merge_frames() -> List[Path]:
    ensure_dir(MERGED_ROOT)
    merged: List[Path] = []

    for ds in DATASET_IDS:
        root = RAW_ROOT / ds
        in_h5 = sorted(root.glob(INPUT_H5_PATTERN))
        out_h5 = sorted(root.glob(OUTPUT_H5_PATTERN))
        vtk = {frame_id(p): p for p in sorted(root.glob(VTK_PATTERN))}

        in_map = {frame_id(p): p for p in in_h5}
        out_map = {frame_id(p): p for p in out_h5}
        common = sorted(set(in_map).intersection(out_map))
        print(f"[merge] {ds}: input={len(in_h5)} output={len(out_h5)} paired={len(common)}")

        out_dir = ensure_dir(MERGED_ROOT / ds)
        for fr in common:
            pin = in_map[fr]
            pout = out_map[fr]
            payload = {}
            payload.update(read_h5_selected(pin, INPUT_KEYS))
            payload.update(read_h5_selected(pout, OUTPUT_KEYS))
            payload["source_dataset"] = np.asarray(ds, dtype=object)
            payload["frame_id"] = np.asarray(fr, dtype=object)
            payload["source_vtk_path"] = np.asarray(str(vtk.get(fr, "")), dtype=object)
            payload["source_input_h5_path"] = np.asarray(str(pin), dtype=object)
            payload["source_output_h5_path"] = np.asarray(str(pout), dtype=object)

            op = out_dir / f"{ds}__frame_{fr}.npz"
            np.savez_compressed(op, **payload)
            merged.append(op)

    merged = sorted(merged)
    print("[merge] total merged:", len(merged))
    return merged


# ==============================================================================
# Task-2 (optional, unchanged style)
# ==============================================================================

def build_grid(bounds, res):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = res
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xg, yg, zg], axis=-1)


def project_nearest(pxyz: np.ndarray, values: np.ndarray, grid_xyz: np.ndarray):
    flat = grid_xyz.reshape(-1, 3)
    if SCIPY_AVAILABLE:
        tree = cKDTree(flat)
        _, j = tree.query(pxyz, k=1)
    else:
        diff = pxyz[:, None, :] - flat[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        j = np.argmin(d2, axis=1)

    m = flat.shape[0]
    c = values.shape[1]
    acc = np.zeros((m, c), dtype=np.float64)
    cnt = np.zeros((m, 1), dtype=np.float64)
    for i, k in enumerate(j):
        acc[int(k)] += values[i]
        cnt[int(k)] += 1.0
    out = acc / np.maximum(cnt, 1.0)
    nx, ny, nz, _ = grid_xyz.shape
    return out.reshape(nx, ny, nz, c).astype(np.float32)


def compute_bounds(merged: List[Path]):
    mn = np.array([np.inf, np.inf, np.inf])
    mx = np.array([-np.inf, -np.inf, -np.inf])
    for p in merged:
        with np.load(p, allow_pickle=True) as d:
            xyz = as_xyz(d["particle_xyz"])
        mn = np.minimum(mn, xyz.min(axis=0))
        mx = np.maximum(mx, xyz.max(axis=0))
    span = np.maximum(mx - mn, 1e-9)
    pad = 0.05 * span
    mn -= pad
    mx += pad
    return float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2])


def _channel_norm_grid(x: np.ndarray, train_idx: np.ndarray):
    axes = tuple(i for i in range(x.ndim) if i != 1)
    mean = np.mean(x[train_idx], axis=axes, keepdims=True)
    std = np.std(x[train_idx], axis=axes, keepdims=True)
    std = np.maximum(std, 1e-8)
    return mean.astype(np.float32), std.astype(np.float32), ((x - mean) / std).astype(np.float32)


def _split_idx(n: int, seed: int = 42, train: float = 0.7, val: float = 0.15):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    nt = int(round(train * n))
    nv = int(round(val * n))
    tr = np.sort(p[:nt])
    va = np.sort(p[nt : nt + nv])
    te = np.sort(p[nt + nv :])
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def build_field_dataset(merged: List[Path]) -> Path:
    bounds = compute_bounds(merged)
    grid = build_grid(bounds, GRID_RESOLUTION)

    Xs, Ys = [], []
    cases, frames = [], []

    for p in merged:
        with np.load(p, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}

        xyz = as_xyz(data["particle_xyz"])
        gamma = as_xyz(data["Gamma_vec"])
        n = xyz.shape[0]
        sigma = _fit_scalar(data["sigma"], n, 1e-2)
        vol = _fit_scalar(data["vol"], n, 1.0)
        density = vol / max(float(np.mean(vol)), 1e-12)

        chans = {
            "Gamma_x": gamma[:, 0],
            "Gamma_y": gamma[:, 1],
            "Gamma_z": gamma[:, 2],
            "sigma": sigma,
            "density": density,
        }
        coord = {"X": grid[..., 0], "Y": grid[..., 1], "Z": grid[..., 2]}

        projected = {}
        for k, v in chans.items():
            g = project_nearest(xyz, v.reshape(-1, 1), grid)
            projected[k] = g[..., 0]

        merged_channels = {**projected, **coord}
        channel_order = [c for c in FIELD_INPUT_CHANNELS if c in merged_channels]
        x = np.stack([merged_channels[c] for c in channel_order], axis=0).astype(np.float32)

        src_pts = as_xyz(data["points"]) if "points" in data else None
        yparts = []
        if OUTPUT_MODE in ("U", "UW"):
            U = as_vec_field(data["U"])
            if U.ndim == 4 and U.shape[:3] == GRID_RESOLUTION:
                yparts.append(np.moveaxis(U, -1, 0))
            elif U.ndim == 2 and src_pts is not None and U.shape[0] == src_pts.shape[0]:
                Ug = project_nearest(src_pts, U, grid)
                yparts.append(np.moveaxis(Ug, -1, 0))
            else:
                raise RuntimeError(f"Cannot map U shape {U.shape}")

        if OUTPUT_MODE in ("W", "UW"):
            W = as_vec_field(data["W"])
            if W.ndim == 4 and W.shape[:3] == GRID_RESOLUTION:
                yparts.append(np.moveaxis(W, -1, 0))
            elif W.ndim == 2 and src_pts is not None and W.shape[0] == src_pts.shape[0]:
                Wg = project_nearest(src_pts, W, grid)
                yparts.append(np.moveaxis(Wg, -1, 0))
            else:
                raise RuntimeError(f"Cannot map W shape {W.shape}")

        y = np.concatenate(yparts, axis=0).astype(np.float32)
        Xs.append(x)
        Ys.append(y)
        cases.append(str(np.asarray(data["source_dataset"]).reshape(-1)[0]))
        frames.append(str(np.asarray(data["frame_id"]).reshape(-1)[0]))

    X = np.stack(Xs).astype(np.float32)
    Y = np.stack(Ys).astype(np.float32)

    tr, va, te = _split_idx(len(X), RANDOM_SEED)
    in_mean, in_std, Xn = _channel_norm_grid(X, tr)
    out_mean, out_std, Yn = _channel_norm_grid(Y, tr)

    target_channels = ["U_x", "U_y", "U_z"] if OUTPUT_MODE == "U" else ["W_x", "W_y", "W_z"]
    if OUTPUT_MODE == "UW":
        target_channels = ["U_x", "U_y", "U_z", "W_x", "W_y", "W_z"]

    out_path = OUT_ROOT / "field_dataset.npz"
    np.savez_compressed(
        out_path,
        inputs=X,
        targets=Y,
        inputs_norm=Xn,
        targets_norm=Yn,
        input_channels=np.asarray(FIELD_INPUT_CHANNELS, dtype=object),
        target_channels=np.asarray(target_channels, dtype=object),
        grid_xyz=grid.astype(np.float32),
        bounds=np.asarray(bounds, dtype=np.float32),
        case_names=np.asarray(cases, dtype=object),
        frame_ids=np.asarray(frames, dtype=object),
        train_idx=tr,
        val_idx=va,
        test_idx=te,
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
    )
    print("[field] wrote", out_path, "shape", X.shape, Y.shape)
    return out_path


# ==============================================================================
# Task-1 preprocessing: EVOLUTION DATASET (x_t -> delta_state)
# ==============================================================================

def _state_from_frame(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    xyz = as_xyz(data["particle_xyz"])
    gamma = as_xyz(data["Gamma_vec"])
    n = xyz.shape[0]
    sigma = _fit_scalar(data["sigma"], n, 1e-2)

    return {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "Gamma_x": gamma[:, 0],
        "Gamma_y": gamma[:, 1],
        "Gamma_z": gamma[:, 2],
        "sigma": sigma,
    }


def _feature_matrix_from_state(
    state: Dict[str, np.ndarray],
    n: int,
    phase: float,
    aoa_deg: float,
    freestream: np.ndarray,
) -> np.ndarray:
    feat = {
        "x": state["x"][:n],
        "y": state["y"][:n],
        "z": state["z"][:n],
        "Gamma_x": state["Gamma_x"][:n],
        "Gamma_y": state["Gamma_y"][:n],
        "Gamma_z": state["Gamma_z"][:n],
        "sigma": state["sigma"][:n],
    }
    # Optional context features are only added if requested.
    if "phase" in PARTICLE_INPUT_FEATURES:
        feat["phase"] = np.full(n, phase, dtype=np.float64)
    if "angle_of_attack" in PARTICLE_INPUT_FEATURES:
        feat["angle_of_attack"] = np.full(n, aoa_deg, dtype=np.float64)
    if "freestream_x" in PARTICLE_INPUT_FEATURES:
        feat["freestream_x"] = np.full(n, float(freestream[0]), dtype=np.float64)
    if "freestream_y" in PARTICLE_INPUT_FEATURES:
        feat["freestream_y"] = np.full(n, float(freestream[1]), dtype=np.float64)
    if "freestream_z" in PARTICLE_INPUT_FEATURES:
        feat["freestream_z"] = np.full(n, float(freestream[2]), dtype=np.float64)
    return np.stack([feat[k] for k in PARTICLE_INPUT_FEATURES], axis=1).astype(np.float32)


def _state_matrix(state: Dict[str, np.ndarray], n: int) -> np.ndarray:
    return np.stack([state[k][:n] for k in STATE_NAMES], axis=1).astype(np.float32)


def _normalize_channels_rows(x: np.ndarray, train_rows: np.ndarray):
    mean = np.mean(x[train_rows], axis=0, keepdims=True)
    std = np.std(x[train_rows], axis=0, keepdims=True)
    std = np.maximum(std, 1e-8)
    xn = ((x - mean) / std).astype(np.float32)
    return mean.astype(np.float32), std.astype(np.float32), xn


def _case_split_label(case: str) -> str:
    if case in TRAIN_CASES:
        return "train"
    if case in VAL_CASES:
        return "val"
    if case in TEST_CASES:
        return "test"
    raise ValueError(f"Case {case} is not assigned in TRAIN_CASES/VAL_CASES/TEST_CASES")


def build_particle_evolution_dataset(merged: List[Path]) -> Path:
    by_case: Dict[str, List[Path]] = {}
    for p in merged:
        by_case.setdefault(p.parent.name, []).append(p)
    for c in by_case:
        by_case[c] = sorted(by_case[c], key=lambda x: frame_id(x))

    all_cases = sorted(by_case.keys())
    _validate_case_split(all_cases)

    # Pre-load all frame states per case
    case_frames: Dict[str, List[Dict[str, object]]] = {}
    for case in all_cases:
        meta = _case_meta(case)
        plist = by_case[case]
        T = len(plist)
        if T < 2:
            print(f"[warn] case={case} has <2 frames; skipping")
            continue

        fr_list: List[Dict[str, object]] = []
        for i, p in enumerate(plist):
            with np.load(p, allow_pickle=True) as d:
                data = {k: d[k] for k in d.files}
            state = _state_from_frame(data)
            fr = str(np.asarray(data["frame_id"]).reshape(-1)[0])
            phase = 0.0 if T <= 1 else float(i) / float(T - 1)
            fr_list.append(
                {
                    "frame_id": fr,
                    "state": state,
                    "phase": phase,
                    "path": str(p),
                }
            )
        case_frames[case] = fr_list

    rows_x: List[np.ndarray] = []
    rows_delta: List[np.ndarray] = []
    rows_next: List[np.ndarray] = []

    pair_ranges: List[Tuple] = []
    pair_contexts: List[Dict[str, object]] = []

    # rollout ground-truth container (case-wise sequence of aligned states)
    rollout_cases: List[str] = []
    rollout_true_states: List[np.ndarray] = []
    rollout_phases: List[np.ndarray] = []
    rollout_dts: List[float] = []

    start = 0
    for case in sorted(case_frames.keys()):
        meta = _case_meta(case)
        fr_list = case_frames[case]
        T = len(fr_list)

        # Build rollout ground truth by consistent truncation to minimum particle count across full case.
        nmin_case = min(len(fr["state"]["x"]) for fr in fr_list)
        true_seq = []
        phase_seq = []
        for fr in fr_list:
            st = fr["state"]
            true_seq.append(_state_matrix(st, nmin_case))
            phase_seq.append(float(fr["phase"]))

        rollout_cases.append(case)
        rollout_true_states.append(np.stack(true_seq, axis=0).astype(np.float32))  # [T, Nmin, 7]
        rollout_phases.append(np.asarray(phase_seq, dtype=np.float32))
        rollout_dts.append(float(meta["dt"]))

        # Build consecutive training pairs: x_t -> delta_t.
        for i in range(T - 1):
            curr = fr_list[i]
            nxt = fr_list[i + 1]

            s0 = curr["state"]
            s1 = nxt["state"]

            n0 = len(s0["x"])
            n1 = len(s1["x"])
            n = min(n0, n1)
            if n <= 0:
                continue

            x_feat = _feature_matrix_from_state(
                state=s0,
                n=n,
                phase=float(curr["phase"]),
                aoa_deg=float(meta["aoa_deg"]),
                freestream=np.asarray(meta["freestream"], dtype=np.float64),
            )

            st0 = _state_matrix(s0, n)
            st1 = _state_matrix(s1, n)
            delta = st1 - st0

            end = start + n
            pair_ranges.append((case, curr["frame_id"], nxt["frame_id"], start, end, n))
            pair_contexts.append(
                {
                    "case": case,
                    "frame_t": curr["frame_id"],
                    "frame_tp1": nxt["frame_id"],
                    "start": start,
                    "end": end,
                    "n_particles": n,
                    "phase_t": float(curr["phase"]),
                    "phase_tp1": float(nxt["phase"]),
                    "aoa_deg": float(meta["aoa_deg"]),
                    "freestream": [float(v) for v in np.asarray(meta["freestream"]).reshape(-1)],
                    "dt": float(meta["dt"]),
                }
            )

            rows_x.append(x_feat.astype(np.float32))
            rows_delta.append(delta.astype(np.float32))
            rows_next.append(st1.astype(np.float32))
            start = end

    if not rows_x:
        raise RuntimeError("No Task-1 pairs were built. Check frame availability and metadata.")

    X = np.concatenate(rows_x, axis=0).astype(np.float32)
    Y_delta = np.concatenate(rows_delta, axis=0).astype(np.float32)
    Y_next = np.concatenate(rows_next, axis=0).astype(np.float32)

    # Case-based split at pair level
    pair_split_train = np.array([i for i, r in enumerate(pair_ranges) if _case_split_label(r[0]) == "train"], dtype=np.int64)
    pair_split_val = np.array([i for i, r in enumerate(pair_ranges) if _case_split_label(r[0]) == "val"], dtype=np.int64)
    pair_split_test = np.array([i for i, r in enumerate(pair_ranges) if _case_split_label(r[0]) == "test"], dtype=np.int64)

    train_rows = _rows_from_pair_ids(pair_ranges, pair_split_train)
    val_rows = _rows_from_pair_ids(pair_ranges, pair_split_val)
    test_rows = _rows_from_pair_ids(pair_ranges, pair_split_test)

    # Normalize per feature using TRAIN rows only.
    in_mean, in_std, Xn = _normalize_channels_rows(X, train_rows)
    out_mean, out_std, Yn_delta = _normalize_channels_rows(Y_delta, train_rows)

    # Next-state normalization for optional analysis (not primary training target).
    next_mean = np.mean(Y_next[train_rows], axis=0, keepdims=True)
    next_std = np.maximum(np.std(Y_next[train_rows], axis=0, keepdims=True), 1e-8)
    Yn_next = ((Y_next - next_mean) / next_std).astype(np.float32)

    out_path = OUT_ROOT / "particle_evolution_dataset.npz"
    np.savez_compressed(
        out_path,
        # core supervised data
        inputs_t=X,
        targets_delta=Y_delta,
        targets_next_state=Y_next,
        inputs_t_norm=Xn,
        targets_delta_norm=Yn_delta,
        targets_next_state_norm=Yn_next,
        # naming
        feature_names=np.asarray(PARTICLE_INPUT_FEATURES, dtype=object),
        state_names=np.asarray(STATE_NAMES, dtype=object),
        target_names=np.asarray(TARGET_DELTA_NAMES, dtype=object),
        # pair indexing
        pair_ranges=np.asarray(pair_ranges, dtype=object),
        pair_contexts=np.asarray(pair_contexts, dtype=object),
        train_pair_ids=pair_split_train,
        val_pair_ids=pair_split_val,
        test_pair_ids=pair_split_test,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        # rollout ground truth
        rollout_cases=np.asarray(rollout_cases, dtype=object),
        rollout_true_states=np.asarray(rollout_true_states, dtype=object),
        rollout_phases=np.asarray(rollout_phases, dtype=object),
        rollout_dts=np.asarray(rollout_dts, dtype=np.float32),
        # split metadata
        train_cases=np.asarray(TRAIN_CASES, dtype=object),
        val_cases=np.asarray(VAL_CASES, dtype=object),
        test_cases=np.asarray(TEST_CASES, dtype=object),
        case_metadata=np.asarray(CASE_METADATA, dtype=object),
        # normalization
        in_mean=in_mean.astype(np.float32),
        in_std=in_std.astype(np.float32),
        out_mean=out_mean.astype(np.float32),
        out_std=out_std.astype(np.float32),
        next_mean=next_mean.astype(np.float32),
        next_std=next_std.astype(np.float32),
    )

    # Rich console summary for sanity checks.
    print("\n[task1] Particle evolution dataset built")
    print("  inputs_t shape            :", X.shape)
    print("  targets_delta shape       :", Y_delta.shape)
    print("  targets_next_state shape  :", Y_next.shape)
    print("  n_pairs                   :", len(pair_ranges))
    print("  n_rollout_cases           :", len(rollout_cases))
    print("  feature_names             :", PARTICLE_INPUT_FEATURES)
    print("  target_names              :", TARGET_DELTA_NAMES)
    print("  split(train/val/test) pairs:", len(pair_split_train), len(pair_split_val), len(pair_split_test))
    print("  split(train/val/test) rows :", len(train_rows), len(val_rows), len(test_rows))
    print("  train/val/test cases      :", TRAIN_CASES, VAL_CASES, TEST_CASES)

    return out_path


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    ensure_dir(OUT_ROOT)
    ensure_dir(MERGED_ROOT)

    merged = merge_frames()

    field_path = None
    if RUN_TASK2_FIELD:
        field_path = build_field_dataset(merged)

    particle_path = build_particle_evolution_dataset(merged)

    summary = {
        "raw_root": str(RAW_ROOT),
        "datasets": DATASET_IDS,
        "n_merged_frames": len(merged),
        "task1_particle_evolution_dataset": str(particle_path),
        "task2_field_dataset": None if field_path is None else str(field_path),
        "run_task2_field": RUN_TASK2_FIELD,
        "train_cases": TRAIN_CASES,
        "val_cases": VAL_CASES,
        "test_cases": TEST_CASES,
    }
    (OUT_ROOT / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done] preprocess summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
