"""
Task-1:
- Build particle evolution dataset using consecutive frame pairs.
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

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except Exception:
    PYVISTA_AVAILABLE = False


# ==============================================================================
# USER SETTINGS
# ==============================================================================
RAW_ROOT = Path("/media/dysco/New Volume/Neeraj/neuralop/data/raw data")
DATASET_IDS = ["1", "2", "7", "8", "9", "10", "11"]

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
TASK1_TARGET_MODE = "ugradu"  # "delta" | "ugradu" | "both"

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
TRAIN_CASES = ["1", "2","10","11"]
VAL_CASES = ["8","7"]
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
TARGET_UGRADU_NAMES = [
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "gradUx_x",
    "gradUx_y",
    "gradUx_z",
    "gradUy_x",
    "gradUy_y",
    "gradUy_z",
    "gradUz_x",
    "gradUz_y",
    "gradUz_z",
]

# Geometry-aware particle channels (from per-frame VTK).
USE_GEOMETRY_CHANNELS = True
GEOMETRY_NEAR_THRESHOLD = 0.02
GEOMETRY_CHANNEL_NAMES = [
    "geom_dist",
    "geom_nx",
    "geom_ny",
    "geom_nz",
    "geom_body_near",
]

# Explicit per-frame conditioning channels appended to each particle feature.
# c_n = [alpha_n, U_inf,n, phi_n, geometry embedding]
USE_EXPLICIT_CONDITIONING = True
CONDITIONING_CHANNEL_NAMES = [
    "angle_of_attack",
    "freestream_x",
    "freestream_y",
    "freestream_z",
    "phase",
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
if USE_GEOMETRY_CHANNELS:
    PARTICLE_INPUT_FEATURES = PARTICLE_INPUT_FEATURES + GEOMETRY_CHANNEL_NAMES
if USE_EXPLICIT_CONDITIONING:
    PARTICLE_INPUT_FEATURES = PARTICLE_INPUT_FEATURES + CONDITIONING_CHANNEL_NAMES


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

    if aoa is None:
        aoa = 0.0
    if dt is None:
        dt = 1.0
    if fs is None:
        fs = [0.0, 0.0, 0.0]

    fs_arr = np.asarray(fs, dtype=np.float64).reshape(-1)
    if fs_arr.shape[0] != 3:
        fs_arr = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    return {
        "aoa_deg": float(aoa),
        "freestream": fs_arr,
        "dt": float(dt),
    }


_VTK_GEOM_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, object]] = {}
_GEOM_BACKEND_NOTICE_PRINTED = False


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize row-vectors safely."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def _point_normals_from_cells(points: np.ndarray, cells: List[np.ndarray]) -> np.ndarray:
    """Estimate point normals by averaging adjacent cell normals."""
    normals = np.zeros_like(points, dtype=np.float64)
    for c in cells:
        if c.size < 3:
            continue
        p0 = points[int(c[0])]
        p1 = points[int(c[1])]
        p2 = points[int(c[2])]
        n = np.cross(p1 - p0, p2 - p0)
        mag = np.linalg.norm(n)
        if mag <= 1e-12:
            continue
        n = n / mag
        normals[c.astype(np.int64)] += n

    nz = np.linalg.norm(normals, axis=1) > 1e-12
    if np.any(nz):
        normals[nz] = _normalize_rows(normals[nz])
    return normals


def _load_legacy_vtk_ascii(vp: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read VTK legacy ASCII points/cells and estimate point normals.

    This is a dependency-light fallback for environments where `pyvista`
    is unavailable. It is sufficient for common FLOWUnsteady wing VTK files.
    """
    txt = vp.read_text(errors="ignore").splitlines()

    npts = None
    p0 = None
    for i, ln in enumerate(txt):
        s = ln.strip()
        if s.upper().startswith("POINTS "):
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid POINTS line in {vp}")
            npts = int(parts[1])
            p0 = i + 1
            break
    if npts is None or p0 is None:
        raise ValueError(f"Could not find POINTS section in {vp}")

    vals: List[float] = []
    i = p0
    while i < len(txt) and len(vals) < 3 * npts:
        s = txt[i].strip()
        if s == "" or s.startswith("#"):
            i += 1
            continue
        head = s.split()[0].upper()
        if head in {
            "CELLS",
            "CELL_TYPES",
            "POINT_DATA",
            "CELL_DATA",
            "SCALARS",
            "VECTORS",
            "LOOKUP_TABLE",
            "FIELD",
        }:
            break
        vals.extend(float(x) for x in s.split())
        i += 1
    if len(vals) < 3 * npts:
        raise ValueError(f"Incomplete POINTS data in {vp}")
    pts = np.asarray(vals[: 3 * npts], dtype=np.float64).reshape(npts, 3)

    cells: List[np.ndarray] = []
    for j, ln in enumerate(txt):
        s = ln.strip()
        if s.upper().startswith("CELLS "):
            parts = s.split()
            if len(parts) < 2:
                break
            ncells = int(parts[1])
            k = j + 1
            for _ in range(ncells):
                if k >= len(txt):
                    break
                row = txt[k].strip().split()
                k += 1
                if not row:
                    continue
                m = int(row[0])
                if m <= 0:
                    continue
                idx = np.asarray([int(x) for x in row[1 : 1 + m]], dtype=np.int64)
                if idx.size >= 3:
                    cells.append(idx)
            break

    if len(cells) == 0:
        normals = np.zeros_like(pts)
    else:
        normals = _point_normals_from_cells(pts, cells)
    return pts, normals


def _load_vtk_geom(vtk_path: str):
    """Load/cached VTK points and normals.

    Returns (points, normals, tree_or_none). Normals can be zeros if unavailable.
    """
    if not vtk_path:
        return None, None, None
    if vtk_path in _VTK_GEOM_CACHE:
        return _VTK_GEOM_CACHE[vtk_path]

    vp = Path(vtk_path)
    if not vp.exists():
        _VTK_GEOM_CACHE[vtk_path] = (None, None, None)
        return _VTK_GEOM_CACHE[vtk_path]

    try:
        if PYVISTA_AVAILABLE:
            mesh = pv.read(str(vp))
            pts = np.asarray(mesh.points, dtype=np.float64)
            nrm = None

            for k in ["Normals", "normal", "normals"]:
                if hasattr(mesh, "point_data") and k in mesh.point_data:
                    cand = np.asarray(mesh.point_data[k], dtype=np.float64)
                    if cand.ndim == 2 and cand.shape[1] == 3 and cand.shape[0] == pts.shape[0]:
                        nrm = cand
                        break

            # If normals are missing in file, estimate from mesh cells.
            if nrm is None or np.linalg.norm(nrm, axis=1).max(initial=0.0) <= 1e-12:
                if hasattr(mesh, "faces") and mesh.faces is not None and mesh.faces.size > 0:
                    faces = mesh.faces
                    cells: List[np.ndarray] = []
                    i = 0
                    while i < len(faces):
                        m = int(faces[i])
                        if m >= 3:
                            cells.append(np.asarray(faces[i + 1 : i + 1 + m], dtype=np.int64))
                        i += 1 + m
                    nrm = _point_normals_from_cells(pts, cells) if len(cells) > 0 else np.zeros_like(pts)
                else:
                    # Last-resort parser for legacy ASCII VTK.
                    pts2, nrm2 = _load_legacy_vtk_ascii(vp)
                    pts, nrm = pts2, nrm2
        else:
            pts, nrm = _load_legacy_vtk_ascii(vp)

        tree = cKDTree(pts) if (SCIPY_AVAILABLE and pts is not None and pts.size > 0) else None
        _VTK_GEOM_CACHE[vtk_path] = (pts, nrm, tree)
    except Exception as e:
        # Keep running, but expose a clear warning once so the user can fix setup.
        global _GEOM_BACKEND_NOTICE_PRINTED
        if not _GEOM_BACKEND_NOTICE_PRINTED:
            print(f"[geom] warning: failed to load VTK geometry ({e}). Geometry channels will be zeros for affected frames.")
            _GEOM_BACKEND_NOTICE_PRINTED = True
        _VTK_GEOM_CACHE[vtk_path] = (None, None, None)

    return _VTK_GEOM_CACHE[vtk_path]


def _particle_geometry_features(xyz: np.ndarray, vtk_path: str, n: int) -> Dict[str, np.ndarray]:
    """Compute per-particle geometry channels by nearest surface query."""
    zeros = np.zeros(n, dtype=np.float64)
    if not USE_GEOMETRY_CHANNELS:
        return {
            "geom_dist": zeros,
            "geom_nx": zeros,
            "geom_ny": zeros,
            "geom_nz": zeros,
            "geom_body_near": zeros,
        }

    pts, nrm, tree = _load_vtk_geom(vtk_path)
    if pts is None or nrm is None or len(pts) == 0:
        return {
            "geom_dist": zeros,
            "geom_nx": zeros,
            "geom_ny": zeros,
            "geom_nz": zeros,
            "geom_body_near": zeros,
        }

    q = xyz[:n]
    if tree is not None:
        dist, idx = tree.query(q, k=1)
    else:
        diff = q[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(d2, axis=1)
        dist = np.sqrt(np.min(d2, axis=1))

    nn = nrm[idx]
    body_near = (dist <= GEOMETRY_NEAR_THRESHOLD).astype(np.float64)
    return {
        "geom_dist": np.asarray(dist, dtype=np.float64),
        "geom_nx": np.asarray(nn[:, 0], dtype=np.float64),
        "geom_ny": np.asarray(nn[:, 1], dtype=np.float64),
        "geom_nz": np.asarray(nn[:, 2], dtype=np.float64),
        "geom_body_near": body_near,
    }


def _print_geometry_channel_stats(X: np.ndarray, feature_names: List[str], tag: str) -> None:
    """Print compact geometry-channel statistics for sanity checks."""
    if not USE_GEOMETRY_CHANNELS:
        print(f"[{tag}] geometry channels disabled.")
        return
    print(f"[{tag}] geometry channel stats:")
    all_zero = True
    for k in GEOMETRY_CHANNEL_NAMES:
        if k not in feature_names:
            print(f"  - {k}: missing from feature_names")
            continue
        j = feature_names.index(k)
        v = X[:, j]
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        vmean = float(np.mean(v))
        nnz = int(np.count_nonzero(v))
        print(f"  - {k:14s} min={vmin:.6g} max={vmax:.6g} mean={vmean:.6g} nnz={nnz}")
        if nnz > 0:
            all_zero = False
    if all_zero:
        print(
            f"[{tag}] warning: all geometry channels are zero. "
            "Check VTK paths, VTK parser backend, and GEOMETRY_NEAR_THRESHOLD."
        )


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
    geom_feat: Dict[str, np.ndarray] = None,
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
    if geom_feat is not None:
        for gk in GEOMETRY_CHANNEL_NAMES:
            if gk in PARTICLE_INPUT_FEATURES:
                gv = geom_feat.get(gk, None)
                if gv is None:
                    feat[gk] = np.zeros(n, dtype=np.float64)
                else:
                    feat[gk] = np.asarray(gv[:n], dtype=np.float64)
    else:
        for gk in GEOMETRY_CHANNEL_NAMES:
            if gk in PARTICLE_INPUT_FEATURES:
                feat[gk] = np.zeros(n, dtype=np.float64)
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
            vtk_path = str(np.asarray(data.get("source_vtk_path", "")).reshape(-1)[0])
            phase = 0.0 if T <= 1 else float(i) / float(T - 1)
            fr_list.append(
                {
                    "frame_id": fr,
                    "state": state,
                    "phase": phase,
                    "path": str(p),
                    "vtk_path": vtk_path,
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

            curr_xyz = np.stack([s0["x"][:n], s0["y"][:n], s0["z"][:n]], axis=1)
            geom_feat = _particle_geometry_features(curr_xyz, str(curr.get("vtk_path", "")), n)

            x_feat = _feature_matrix_from_state(
                state=s0,
                n=n,
                phase=float(curr["phase"]),
                aoa_deg=float(meta["aoa_deg"]),
                freestream=np.asarray(meta["freestream"], dtype=np.float64),
                geom_feat=geom_feat,
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
        use_geometry_channels=np.asarray(USE_GEOMETRY_CHANNELS),
        geometry_channel_names=np.asarray(GEOMETRY_CHANNEL_NAMES, dtype=object),
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
    print("  use_geometry_channels     :", USE_GEOMETRY_CHANNELS)
    print("  target_names              :", TARGET_DELTA_NAMES)
    print("  split(train/val/test) pairs:", len(pair_split_train), len(pair_split_val), len(pair_split_test))
    print("  split(train/val/test) rows :", len(train_rows), len(val_rows), len(test_rows))
    print("  train/val/test cases      :", TRAIN_CASES, VAL_CASES, TEST_CASES)
    _print_geometry_channel_stats(X, PARTICLE_INPUT_FEATURES, "task1-delta")

    return out_path


def build_particle_ugradu_dataset(merged: List[Path]) -> Path:
    """Build instantaneous Task-1 dataset: x_t -> [u_t, gradU_t].

    This is the surrogate mode where FLOWUnsteady still performs particle
    integration, while the ML model replaces only the expensive U/gradU query.
    """
    by_case: Dict[str, List[Path]] = {}
    for p in merged:
        by_case.setdefault(p.parent.name, []).append(p)
    for c in by_case:
        by_case[c] = sorted(by_case[c], key=lambda x: frame_id(x))

    all_cases = sorted(by_case.keys())
    _validate_case_split(all_cases)

    # Keep each frame as one coherent graph sample.
    frame_inputs_raw: List[np.ndarray] = []
    frame_targets_raw: List[np.ndarray] = []
    frame_ranges: List[Tuple] = []
    frame_contexts: List[Dict[str, object]] = []

    start = 0
    for case in all_cases:
        meta = _case_meta(case)
        fr_list = by_case[case]
        T = len(fr_list)

        for i, p in enumerate(fr_list):
            with np.load(p, allow_pickle=True) as d:
                data = {k: d[k] for k in d.files}

            state = _state_from_frame(data)
            xyz = as_xyz(data["particle_xyz"])
            vel = as_xyz(data["velocity"])
            gx = as_xyz(data["velocity_gradient_x"])
            gy = as_xyz(data["velocity_gradient_y"])
            gz = as_xyz(data["velocity_gradient_z"])
            grad = np.concatenate([gx, gy, gz], axis=1)

            n = min(
                xyz.shape[0],
                vel.shape[0],
                gx.shape[0],
                gy.shape[0],
                gz.shape[0],
            )
            if n <= 0:
                continue

            phase = 0.0 if T <= 1 else float(i) / float(T - 1)
            vtk_path = str(np.asarray(data.get("source_vtk_path", "")).reshape(-1)[0])
            geom_feat = _particle_geometry_features(xyz, vtk_path, n)
            x_feat = _feature_matrix_from_state(
                state=state,
                n=n,
                phase=phase,
                aoa_deg=float(meta["aoa_deg"]),
                freestream=np.asarray(meta["freestream"], dtype=np.float64),
                geom_feat=geom_feat,
            )
            y = np.concatenate([vel[:n], grad[:n]], axis=1).astype(np.float32)

            end = start + n
            fr = str(np.asarray(data["frame_id"]).reshape(-1)[0])
            frame_ranges.append((case, fr, start, end, n))
            frame_contexts.append(
                {
                    "case": case,
                    "frame": fr,
                    "start": start,
                    "end": end,
                    "n_particles": n,
                    "phase": phase,
                    "aoa_deg": float(meta["aoa_deg"]),
                    "freestream": [float(v) for v in np.asarray(meta["freestream"]).reshape(-1)],
                    "dt": float(meta["dt"]),
                    "vtk_path": vtk_path,
                }
            )

            x_this = x_feat.astype(np.float32)
            y_this = y.astype(np.float32)
            frame_inputs_raw.append(x_this)
            frame_targets_raw.append(y_this)

            start = end

    if not frame_inputs_raw:
        raise RuntimeError("No Task-1 u/gradU samples were built")

    frame_ids_train = np.array([i for i, r in enumerate(frame_ranges) if _case_split_label(r[0]) == "train"], dtype=np.int64)
    frame_ids_val = np.array([i for i, r in enumerate(frame_ranges) if _case_split_label(r[0]) == "val"], dtype=np.int64)
    frame_ids_test = np.array([i for i, r in enumerate(frame_ranges) if _case_split_label(r[0]) == "test"], dtype=np.int64)

    def rows_from_frame_ids(fid: np.ndarray) -> np.ndarray:
        chunks = []
        for i in fid:
            _, _, s, e, _ = frame_ranges[int(i)]
            chunks.append(np.arange(int(s), int(e), dtype=np.int64))
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.int64)

    train_rows = rows_from_frame_ids(frame_ids_train)
    val_rows = rows_from_frame_ids(frame_ids_val)
    test_rows = rows_from_frame_ids(frame_ids_test)

    # Normalize using ONLY training-frame particles.
    X_train = np.concatenate([frame_inputs_raw[int(i)] for i in frame_ids_train], axis=0).astype(np.float32)
    Y_train = np.concatenate([frame_targets_raw[int(i)] for i in frame_ids_train], axis=0).astype(np.float32)
    in_mean = np.mean(X_train, axis=0, keepdims=True)
    in_std = np.maximum(np.std(X_train, axis=0, keepdims=True), 1e-8)
    out_mean = np.mean(Y_train, axis=0, keepdims=True)
    out_std = np.maximum(np.std(Y_train, axis=0, keepdims=True), 1e-8)

    frame_inputs_norm = [((x - in_mean) / in_std).astype(np.float32) for x in frame_inputs_raw]
    frame_targets_norm = [((y - out_mean) / out_std).astype(np.float32) for y in frame_targets_raw]

    # Flat arrays kept only as optional legacy compatibility keys.
    X = np.concatenate(frame_inputs_raw, axis=0).astype(np.float32)
    Y = np.concatenate(frame_targets_raw, axis=0).astype(np.float32)
    Xn = np.concatenate(frame_inputs_norm, axis=0).astype(np.float32)
    Yn = np.concatenate(frame_targets_norm, axis=0).astype(np.float32)

    out_path = OUT_ROOT / "particle_ugradu_dataset.npz"
    np.savez_compressed(
        out_path,
        # Frame-graph dataset (preferred)
        inputs_by_frame=np.asarray(frame_inputs_raw, dtype=object),
        targets_by_frame=np.asarray(frame_targets_raw, dtype=object),
        inputs_by_frame_norm=np.asarray(frame_inputs_norm, dtype=object),
        targets_by_frame_norm=np.asarray(frame_targets_norm, dtype=object),
        # Flat legacy arrays (optional compatibility with older scripts)
        inputs_t=X,
        targets_ugradu=Y,
        inputs_t_norm=Xn,
        targets_ugradu_norm=Yn,
        feature_names=np.asarray(PARTICLE_INPUT_FEATURES, dtype=object),
        target_names=np.asarray(TARGET_UGRADU_NAMES, dtype=object),
        frame_ranges=np.asarray(frame_ranges, dtype=object),
        frame_contexts=np.asarray(frame_contexts, dtype=object),
        train_frame_ids=frame_ids_train,
        val_frame_ids=frame_ids_val,
        test_frame_ids=frame_ids_test,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        train_cases=np.asarray(TRAIN_CASES, dtype=object),
        val_cases=np.asarray(VAL_CASES, dtype=object),
        test_cases=np.asarray(TEST_CASES, dtype=object),
        case_metadata=np.asarray(CASE_METADATA, dtype=object),
        use_explicit_conditioning=np.asarray(USE_EXPLICIT_CONDITIONING),
        conditioning_channel_names=np.asarray(CONDITIONING_CHANNEL_NAMES, dtype=object),
        use_geometry_channels=np.asarray(USE_GEOMETRY_CHANNELS),
        geometry_channel_names=np.asarray(GEOMETRY_CHANNEL_NAMES, dtype=object),
        in_mean=in_mean.astype(np.float32),
        in_std=in_std.astype(np.float32),
        out_mean=out_mean.astype(np.float32),
        out_std=out_std.astype(np.float32),
    )

    print("\n[task1-ugradu] dataset built")
    print("  n_graph_frames            :", len(frame_inputs_raw))
    print("  first graph shapes        :", frame_inputs_raw[0].shape, frame_targets_raw[0].shape)
    print("  inputs_t shape (legacy)   :", X.shape)
    print("  targets_ugradu (legacy)   :", Y.shape)
    print("  n_frames                  :", len(frame_ranges))
    print("  feature_names             :", PARTICLE_INPUT_FEATURES)
    print("  use_explicit_conditioning :", USE_EXPLICIT_CONDITIONING)
    print("  conditioning_channels     :", CONDITIONING_CHANNEL_NAMES if USE_EXPLICIT_CONDITIONING else [])
    print("  use_geometry_channels     :", USE_GEOMETRY_CHANNELS)
    print("  target_names              :", TARGET_UGRADU_NAMES)
    print("  split(train/val/test) rows:", len(train_rows), len(val_rows), len(test_rows))
    print("  split(train/val/test) cases:", TRAIN_CASES, VAL_CASES, TEST_CASES)
    _print_geometry_channel_stats(X, PARTICLE_INPUT_FEATURES, "task1-ugradu")
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

    particle_evolution_path = None
    particle_ugradu_path = None
    mode = str(TASK1_TARGET_MODE).lower()
    if mode == "delta":
        particle_evolution_path = build_particle_evolution_dataset(merged)
    elif mode == "ugradu":
        particle_ugradu_path = build_particle_ugradu_dataset(merged)
    elif mode == "both":
        particle_evolution_path = build_particle_evolution_dataset(merged)
        particle_ugradu_path = build_particle_ugradu_dataset(merged)
    else:
        raise ValueError("TASK1_TARGET_MODE must be one of: delta, ugradu, both")

    summary = {
        "raw_root": str(RAW_ROOT),
        "datasets": DATASET_IDS,
        "n_merged_frames": len(merged),
        "task1_target_mode": mode,
        "task1_particle_evolution_dataset": None if particle_evolution_path is None else str(particle_evolution_path),
        "task1_particle_ugradu_dataset": None if particle_ugradu_path is None else str(particle_ugradu_path),
        "task2_field_dataset": None if field_path is None else str(field_path),
        "run_task2_field": RUN_TASK2_FIELD,
        "use_explicit_conditioning": USE_EXPLICIT_CONDITIONING,
        "conditioning_channel_names": CONDITIONING_CHANNEL_NAMES if USE_EXPLICIT_CONDITIONING else [],
        "use_geometry_channels": USE_GEOMETRY_CHANNELS,
        "geometry_channel_names": GEOMETRY_CHANNEL_NAMES,
        "geometry_near_threshold": GEOMETRY_NEAR_THRESHOLD,
        "pyvista_available": PYVISTA_AVAILABLE,
        "scipy_available": SCIPY_AVAILABLE,
        "train_cases": TRAIN_CASES,
        "val_cases": VAL_CASES,
        "test_cases": TEST_CASES,
    }
    (OUT_ROOT / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done] preprocess summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
