from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    from scipy.spatial import cKDTree

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


BASE_ROOT = Path("/media/dysco/New Volume/Neeraj/neuralop/data")
FIELD_ROOT = BASE_ROOT / "task2"

# Task-2 field files currently contain only Eulerian outputs. The particle
# state is paired from the matching case/frame in Task-1 by default.
PARTICLE_ROOT = Path(os.environ.get("FINAL2_TASK2_PARTICLE_ROOT", str(BASE_ROOT / "task1")))
FIELD_ROOT_CANDIDATES = [
    FIELD_ROOT,
    Path(__file__).resolve().parents[1] / "data" / "task2",
]

OUT_ROOT = Path(__file__).resolve().parent / "processed_data_task2"
TASK2_OUT_ROOT = OUT_ROOT / "task2_gino_frames"

DYNAMIC_PARTICLE_H5_PATTERN = "static_airfoil_pfield.*.h5"
STATIC_PARTICLE_H5_PATTERN = "static_airfoil_staticpfield.*.h5"
FIELD_H5_PATTERN = "static_airfoil_fdom.*.h5"
INCLUDE_STATIC_PARTICLES = True

# CASE_RE = re.compile(r"^(?P<aoa>\d+(?:\.\d+)?)deg_static_airfoil_(?P<speed>\d+(?:\.\d+)?)u_(?P<particles>\d+)p$")
CASE_RE = re.compile(
    r"^(?P<aoa>\d+(?:\.\d+)?)deg_static_airfoil_"
    r"(?P<speed>\d+(?:\.\d+)?)u_"
    r"(?P<particles>\d+)p"
    r"(?:_(?P<variant>tsr|ssr))?$"
)

FRAME_RE = re.compile(r"(\d+)(?!.*\d)")

RANDOM_SEED = 42
DEFAULT_DT = 0.0034

TRAIN_AOA_DEGREES = list(range(10, 31, 2))
VAL_AOA_DEGREES = [11, 15, 21, 25]
TEST_NORMAL_AOA_DEGREES = [27]
TEST_SUPER_RESOLUTION_AOA_DEGREES = [19]
TEST_UNSEEN_AOA_DEGREES = [32]

USE_SAME_DISTRIBUTION_VALIDATION = True
VAL_ID_FRAME_STRIDE = 5
VAL_ID_FRAME_OFFSET = 2
VAL_ID_MIN_FRAMES_PER_TRAIN_CASE = 4

# The raw Task-2 fdom files in the current dataset are 65^3. Use 64^3 for
# training by index-resampling while retaining the full-resolution arrays for
# super-resolution evaluation if desired by the notebook.
TARGET_GRID_RESOLUTION = (64, 64, 64)
SAVE_FULL_FIELD_FOR_SR = False
MAX_CASES = int(os.environ.get("FINAL2_TASK2_MAX_CASES", "0") or "0")
MAX_FRAMES_PER_CASE = int(os.environ.get("FINAL2_TASK2_MAX_FRAMES_PER_CASE", "0") or "0")

# Match Task1-v2 channel selection: keep particle state, useful geometry, AoA,
# and phase; do not keep geom_body_near or deterministic freestream_x/z.
PARTICLE_INPUT_FEATURES = [
    "x",
    "y",
    "z",
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

TARGET_NAMES = ["U_x", "U_y", "U_z", "W_x", "W_y", "W_z"]
MIN_PARTICLES_PER_FRAME = 64


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_id(path: Path) -> str:
    m = FRAME_RE.search(path.stem)
    return m.group(1).zfill(6) if m else path.stem


# def parse_case_name(case: str) -> Optional[Dict[str, float]]:
#     m = CASE_RE.match(str(case))
#     if not m:
#         return None
#     return {
#         "aoa_deg": float(m.group("aoa")),
#         "magVinf": float(m.group("speed")),
#         "particles_per_step": float(m.group("particles")),
#     }

def parse_case_name(case):
    match = CASE_RE.match(case)
    if match is None:
        return None

    return {
        "aoa_deg": float(match.group("aoa")),
        "speed": float(match.group("speed")),
        "particles": int(match.group("particles")),
        "variant": match.group("variant"),
    }

def resolve_field_root() -> Path:
    env_root = os.environ.get("FINAL2_TASK2_FIELD_ROOT", "").strip()
    if env_root:
        p = Path(env_root)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"FINAL2_TASK2_FIELD_ROOT is invalid: {p}")
    for cand in FIELD_ROOT_CANDIDATES:
        if cand.is_dir():
            return cand
    tried = ", ".join(str(p) for p in FIELD_ROOT_CANDIDATES)
    raise FileNotFoundError(f"Could not locate Task-2 field root. Tried: {tried}")


def discover_cases(field_root: Path) -> List[str]:
    cases = []
    for p in sorted(field_root.iterdir()):
        if not p.is_dir() or parse_case_name(p.name) is None:
            continue
        if len(list(p.glob(FIELD_H5_PATTERN))) > 0:
            cases.append(p.name)
    if not cases:
        raise RuntimeError(f"No Task-2 case folders with {FIELD_H5_PATTERN} found in {field_root}")
    return cases


def case_metadata(case: str, n_frames: int, frame_index: int) -> Dict[str, Any]:
    parsed = parse_case_name(case)
    if parsed is None:
        raise ValueError(f"Cannot parse case metadata from {case}")
    aoa = float(parsed["aoa_deg"])
    # mag = float(parsed["magVinf"])
    mag = float(parsed["speed"])
    rad = np.deg2rad(aoa)
    phase = 0.0 if n_frames <= 1 else float(frame_index) / float(n_frames - 1)
    return {
        "aoa_deg": aoa,
        "magVinf": mag,
        "freestream": [mag * np.cos(rad), 0.0, mag * np.sin(rad)],
        "particles_per_step": int(round(parsed["particles"])),
        "dt": DEFAULT_DT,
        "phase": phase,
    }


# def split_role(case: str) -> str:
#     parsed = parse_case_name(case)
#     if parsed is None:
#         return "unassigned"
#     aoa = int(round(parsed["aoa_deg"]))
#     if aoa in TRAIN_AOA_DEGREES:
#         return "train_case"
#     if aoa in VAL_AOA_DEGREES:
#         return "validation_angle"
#     if aoa in TEST_NORMAL_AOA_DEGREES:
#         return "testing_normal"
#     if aoa in TEST_SUPER_RESOLUTION_AOA_DEGREES:
#         return "testing_super_resolution_case"
#     if aoa in TEST_UNSEEN_AOA_DEGREES:
#         return "testing_unseen_angle"
#     return "unassigned"

def split_role(case: str) -> str:
    parsed = parse_case_name(case)
    if parsed is None:
        return "unassigned"

    aoa = int(round(parsed["aoa_deg"]))
    variant = parsed.get("variant", None)

    if aoa in TRAIN_AOA_DEGREES:
        return "train_case"

    if aoa in VAL_AOA_DEGREES:
        return "validation_angle"

    if aoa in TEST_NORMAL_AOA_DEGREES:
        return "testing_normal"

    if aoa in TEST_UNSEEN_AOA_DEGREES:
        return "testing_unseen_angle"

    if variant == "ssr":
        return "testing_spatial_sr"

    if variant == "tsr":
        return "testing_temporal_sr"

    return "unassigned"


def read_h5(path: Path, keys: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as f:
        for key in keys:
            if key not in f:
                raise KeyError(f"{path}: missing {key}")
            out[key] = np.asarray(f[key])
    return out


def as_xyz(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.astype(np.float64)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T.astype(np.float64)
    raise ValueError(f"Cannot parse xyz/vector array from shape={a.shape}")


def fit_scalar(arr: np.ndarray, n: int, default: float) -> np.ndarray:
    x = np.asarray(arr).reshape(-1)
    if x.size == 0:
        return np.full(n, default, dtype=np.float64)
    if x.size < n:
        out = np.full(n, default, dtype=np.float64)
        out[: x.size] = x
        return out
    return x[:n].astype(np.float64)


def concat_payloads(dynamic: Dict[str, np.ndarray], static: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if static is None or not INCLUDE_STATIC_PARTICLES:
        return {k: np.asarray(v) for k, v in dynamic.items()}
    out = {}
    for key in dynamic:
        a = np.asarray(dynamic[key])
        b = np.asarray(static[key])
        out[key] = a if a.ndim == 0 or b.ndim == 0 else np.concatenate([a, b], axis=0)
    return out


def field_grid_from_nodes(nodes: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[int, int, int]]:
    axes = tuple(np.unique(nodes[:, i]) for i in range(3))
    res = tuple(len(a) for a in axes)
    if int(np.prod(res)) != nodes.shape[0]:
        raise RuntimeError(f"Field nodes are not a complete tensor grid: unique={res}, n={nodes.shape[0]}")
    return axes, res


def reshape_field(nodes: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axes, res = field_grid_from_nodes(nodes)
    order = np.lexsort((nodes[:, 2], nodes[:, 1], nodes[:, 0]))
    sorted_nodes = nodes[order]
    sorted_values = values[order]
    grid_xyz = sorted_nodes.reshape(*res, 3)
    field = sorted_values.reshape(*res, values.shape[1])
    return grid_xyz.astype(np.float32), field.astype(np.float32)


def regular_index_resample(arr: np.ndarray, target_res: Tuple[int, int, int]) -> np.ndarray:
    src = arr.shape[:3]
    if tuple(src) == tuple(target_res):
        return arr.astype(np.float32)
    idx = [np.round(np.linspace(0, src[d] - 1, target_res[d])).astype(np.int64) for d in range(3)]
    return arr[np.ix_(idx[0], idx[1], idx[2])].astype(np.float32)


def simple_geometry_features(xyz: np.ndarray) -> Dict[str, np.ndarray]:
    # Dependency-light airfoil/body proxy: distance and outward direction from
    # the particle-cloud center. The task1 notebook removed geom_body_near, so
    # this keeps only continuous channels with useful variation.
    center = np.median(xyz, axis=0, keepdims=True)
    rel = xyz - center
    dist = np.linalg.norm(rel, axis=1)
    scale = np.maximum(dist[:, None], 1e-12)
    normal = rel / scale
    return {
        "geom_dist": dist.astype(np.float64),
        "geom_nx": normal[:, 0].astype(np.float64),
        "geom_ny": normal[:, 1].astype(np.float64),
        "geom_nz": normal[:, 2].astype(np.float64),
    }


def particle_feature_matrix(payload: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
    xyz = as_xyz(payload["X"])
    gamma = as_xyz(payload["Gamma"])
    n = xyz.shape[0]
    sigma = fit_scalar(payload["sigma"], n, 1e-2)
    geom = simple_geometry_features(xyz)
    feature_map = {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "Gamma_x": gamma[:, 0],
        "Gamma_y": gamma[:, 1],
        "Gamma_z": gamma[:, 2],
        "sigma": sigma,
        "geom_dist": geom["geom_dist"],
        "geom_nx": geom["geom_nx"],
        "geom_ny": geom["geom_ny"],
        "geom_nz": geom["geom_nz"],
        "angle_of_attack": np.full(n, float(meta["aoa_deg"]), dtype=np.float64),
        "phase": np.full(n, float(meta["phase"]), dtype=np.float64),
    }
    return np.stack([feature_map[name] for name in PARTICLE_INPUT_FEATURES], axis=1).astype(np.float32)


def normalize_points(xyz: np.ndarray, coord_min: np.ndarray, coord_span: np.ndarray) -> np.ndarray:
    return np.clip((xyz - coord_min) / coord_span, 0.0, 1.0).astype(np.float32)


def compute_channel_stats(sample_paths: List[Path], key: str, train_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s1 = None
    s2 = None
    n = 0
    for idx in train_ids.tolist():
        with np.load(sample_paths[int(idx)], allow_pickle=True) as d:
            arr = np.asarray(d[key], dtype=np.float64).reshape(-1, len(d[f"{key}_names"]) if f"{key}_names" in d else np.asarray(d[key]).shape[-1])
        finite = np.all(np.isfinite(arr), axis=1)
        if not np.all(finite):
            bad = int(arr.shape[0] - np.count_nonzero(finite))
            print(f"[stats] warning: dropping {bad} non-finite rows from {sample_paths[int(idx)].name}:{key}")
            arr = arr[finite]
        if arr.size == 0:
            continue
        if s1 is None:
            s1 = arr.sum(axis=0)
            s2 = (arr * arr).sum(axis=0)
        else:
            s1 += arr.sum(axis=0)
            s2 += (arr * arr).sum(axis=0)
        n += arr.shape[0]
    if s1 is None or n == 0:
        raise RuntimeError(f"No finite rows available to compute stats for key={key}")
    mean = s1 / max(n, 1)
    var = s2 / max(n, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def make_same_distribution_val(frame_contexts: List[Dict[str, Any]], train_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not USE_SAME_DISTRIBUTION_VALIDATION:
        return train_ids, np.zeros((0,), dtype=np.int64)
    by_case: Dict[str, List[int]] = defaultdict(list)
    for i in train_ids.tolist():
        by_case[str(frame_contexts[int(i)]["case"])].append(int(i))
    train_out, val_out = [], []
    for case, ids in sorted(by_case.items()):
        ids = sorted(ids, key=lambda j: int(frame_contexts[j]["frame"]))
        if len(ids) <= 1:
            train_out.extend(ids)
            continue
        stride = max(int(VAL_ID_FRAME_STRIDE), 2)
        offset = min(max(int(VAL_ID_FRAME_OFFSET), 0), stride - 1)
        val_ids = ids[offset::stride]
        min_val = min(max(int(VAL_ID_MIN_FRAMES_PER_TRAIN_CASE), 1), max(len(ids) - 1, 1))
        if len(val_ids) < min_val:
            seen = set(val_ids)
            val_ids.extend([j for j in ids if j not in seen][: min_val - len(val_ids)])
        val_set = set(val_ids)
        tr_ids = [j for j in ids if j not in val_set]
        if not tr_ids:
            tr_ids, val_ids = ids[:-1], ids[-1:]
        train_out.extend(tr_ids)
        val_out.extend(val_ids)
    return np.asarray(sorted(train_out), dtype=np.int64), np.asarray(sorted(val_out), dtype=np.int64)


def build_dataset() -> Path:
    field_root = resolve_field_root()
    if not PARTICLE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Particle root does not exist: {PARTICLE_ROOT}. "
            "Set FINAL2_TASK2_PARTICLE_ROOT if Task-2 particle files live elsewhere."
        )

    ensure_dir(OUT_ROOT)
    ensure_dir(TASK2_OUT_ROOT)

    cases = discover_cases(field_root)
    if MAX_CASES > 0:
        cases = cases[:MAX_CASES]
    sample_paths: List[Path] = []
    contexts: List[Dict[str, Any]] = []
    coord_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    coord_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    particle_keys = ["X", "Gamma", "sigma", "vol", "static"]

    for case in cases:
        role = split_role(case)
        field_files = sorted((field_root / case).glob(FIELD_H5_PATTERN), key=frame_id)
        if MAX_FRAMES_PER_CASE > 0:
            field_files = field_files[:MAX_FRAMES_PER_CASE]
        
        particle_case = case
        # SR field folders reuse the same particle dataset
        if particle_case.endswith("_ssr"):
            particle_case = particle_case[:-4]

        if particle_case.endswith("_tsr"):
            particle_case = particle_case[:-4]
        
        dynamic_map = {frame_id(p): p for p in sorted((PARTICLE_ROOT / particle_case).glob(DYNAMIC_PARTICLE_H5_PATTERN))}

        static_map = {frame_id(p): p for p in sorted((PARTICLE_ROOT / particle_case).glob(STATIC_PARTICLE_H5_PATTERN))}
        if not dynamic_map:
            print(f"[warn] no particle files for case={case}; skipping field outputs")
            continue

        n_written = 0
        for j, fp in enumerate(field_files):
            fr = frame_id(fp)
            pin = dynamic_map.get(fr)
            if pin is None:
                continue
            pstatic = static_map.get(fr)
            dynamic = read_h5(pin, particle_keys)
            static = read_h5(pstatic, particle_keys) if pstatic is not None else None
            payload = concat_payloads(dynamic, static)

            meta = case_metadata(case, len(field_files), j)
            x = particle_feature_matrix(payload, meta)
            if x.shape[0] < MIN_PARTICLES_PER_FRAME:
                continue

            field = read_h5(fp, ["nodes", "U", "W"])
            nodes = as_xyz(field["nodes"])
            U = np.nan_to_num(as_xyz(field["U"]), nan=0.0, posinf=0.0, neginf=0.0)
            W = np.nan_to_num(as_xyz(field["W"]), nan=0.0, posinf=0.0, neginf=0.0)
            grid_xyz_full, U_grid_full = reshape_field(nodes, U)
            _, W_grid_full = reshape_field(nodes, W)

            grid_xyz = regular_index_resample(grid_xyz_full, TARGET_GRID_RESOLUTION)
            U_grid = regular_index_resample(U_grid_full, TARGET_GRID_RESOLUTION)
            W_grid = regular_index_resample(W_grid_full, TARGET_GRID_RESOLUTION)
            y_grid = np.concatenate([U_grid, W_grid], axis=-1).astype(np.float32)

            particle_xyz = x[:, :3].astype(np.float32)
            coord_min = np.minimum(coord_min, np.minimum(particle_xyz.min(axis=0), grid_xyz.reshape(-1, 3).min(axis=0)))
            coord_max = np.maximum(coord_max, np.maximum(particle_xyz.max(axis=0), grid_xyz.reshape(-1, 3).max(axis=0)))

            out_path = TASK2_OUT_ROOT / f"{case}__frame_{fr}.npz"
            save_kwargs = {
                "input_geom": particle_xyz,
                "input_features": x,
                "output_queries": grid_xyz.reshape(-1, 3).astype(np.float32),
                "targets": y_grid.reshape(-1, 6).astype(np.float32),
                "input_feature_names": np.asarray(PARTICLE_INPUT_FEATURES, dtype=object),
                "target_names": np.asarray(TARGET_NAMES, dtype=object),
                "case": np.asarray(case, dtype=object),
                "frame": np.asarray(fr, dtype=object),
                "split_role": np.asarray(role, dtype=object),
                "grid_resolution": np.asarray(TARGET_GRID_RESOLUTION, dtype=np.int64),
                "source_field_path": np.asarray(str(fp), dtype=object),
                "source_particle_path": np.asarray(str(pin), dtype=object),
            }
            if SAVE_FULL_FIELD_FOR_SR:
                save_kwargs["output_queries_full"] = grid_xyz_full.reshape(-1, 3).astype(np.float32)
                save_kwargs["targets_full"] = np.concatenate([U_grid_full, W_grid_full], axis=-1).reshape(-1, 6).astype(np.float32)
                save_kwargs["full_grid_resolution"] = np.asarray(grid_xyz_full.shape[:3], dtype=np.int64)
            np.savez_compressed(out_path, **save_kwargs)
            sample_paths.append(out_path)
            contexts.append(
                {
                    "case": case,
                    "frame": int(fr),
                    "split_role": role,
                    "aoa_deg": float(meta["aoa_deg"]),
                    "phase": float(meta["phase"]),
                    "n_particles": int(x.shape[0]),
                    "n_output_points": int(np.prod(TARGET_GRID_RESOLUTION)),
                }
            )
            n_written += 1
        print(f"[case] {case}: role={role}, fields={len(field_files)}, paired_written={n_written}")

    if not sample_paths:
        raise RuntimeError("No paired Task-2 samples were written. Check Task-2 fields and particle root alignment.")

    train_case_ids = np.asarray([i for i, c in enumerate(contexts) if c["split_role"] == "train_case"], dtype=np.int64)
    train_ids, val_id_ids = make_same_distribution_val(contexts, train_case_ids)
    val_angle_ids = np.asarray([i for i, c in enumerate(contexts) if c["split_role"] == "validation_angle"], dtype=np.int64)
    test_normal_ids = np.asarray([i for i, c in enumerate(contexts) if c["split_role"] == "testing_normal"], dtype=np.int64)
    # test_sr_case_ids = np.asarray([i for i, c in enumerate(contexts) if c["split_role"] == "testing_super_resolution_case"], dtype=np.int64)
    test_spatial_sr_ids = np.asarray(
    [i for i, c in enumerate(contexts)
     if c["split_role"] == "testing_spatial_sr"],
    dtype=np.int64)

    test_temporal_sr_ids = np.asarray(
        [i for i, c in enumerate(contexts)
        if c["split_role"] == "testing_temporal_sr"],
        dtype=np.int64
        )
    test_unseen_ids = np.asarray([i for i, c in enumerate(contexts) if c["split_role"] == "testing_unseen_angle"], dtype=np.int64)

    coord_span = np.maximum(coord_max - coord_min, 1e-12).astype(np.float32)
    coord_min = coord_min.astype(np.float32)

    in_mean, in_std = compute_channel_stats(sample_paths, "input_features", train_ids)
    out_mean, out_std = compute_channel_stats(sample_paths, "targets", train_ids)

    manifest = OUT_ROOT / "task2_gino_dataset.npz"
    np.savez_compressed(
        manifest,
        sample_paths=np.asarray([str(p) for p in sample_paths], dtype=object),
        frame_contexts=np.asarray(contexts, dtype=object),
        feature_names=np.asarray(PARTICLE_INPUT_FEATURES, dtype=object),
        target_names=np.asarray(TARGET_NAMES, dtype=object),
        grid_resolution=np.asarray(TARGET_GRID_RESOLUTION, dtype=np.int64),
        train_frame_ids=train_ids,
        val_id_frame_ids=val_id_ids,
        val_angle_frame_ids=val_angle_ids,
        test_normal_frame_ids=test_normal_ids,
        # test_super_resolution_frame_ids=test_sr_case_ids,
        test_spatial_sr_frame_ids=test_spatial_sr_ids,
        test_temporal_sr_frame_ids=test_temporal_sr_ids,
        test_unseen_angle_frame_ids=test_unseen_ids,
        coord_min=coord_min,
        coord_span=coord_span,
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
        field_root=np.asarray(str(field_root), dtype=object),
        particle_root=np.asarray(str(PARTICLE_ROOT), dtype=object),
    )

    summary = {
        "dataset": str(manifest),
        "n_samples": len(sample_paths),
        "field_root": str(field_root),
        "particle_root": str(PARTICLE_ROOT),
        "feature_names": PARTICLE_INPUT_FEATURES,
        "target_names": TARGET_NAMES,
        "grid_resolution": list(TARGET_GRID_RESOLUTION),
        "split_counts": {
            "train": int(train_ids.size),
            "val_id": int(val_id_ids.size),
            "val_angle": int(val_angle_ids.size),
            "test_normal": int(test_normal_ids.size),
            # "test_super_resolution": int(test_sr_case_ids.size),
            "test_spatial_sr": int(test_spatial_sr_ids.size),
            "test_temporal_sr": int(test_temporal_sr_ids.size),
            "test_unseen_angle": int(test_unseen_ids.size),
        },
        "coord_min": coord_min.tolist(),
        "coord_span": coord_span.tolist(),
        "save_full_field_for_sr": bool(SAVE_FULL_FIELD_FOR_SR),
        "max_cases": int(MAX_CASES),
        "max_frames_per_case": int(MAX_FRAMES_PER_CASE),
    }
    (OUT_ROOT / "preprocess_task2_summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] wrote", manifest)
    print(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    build_dataset()
