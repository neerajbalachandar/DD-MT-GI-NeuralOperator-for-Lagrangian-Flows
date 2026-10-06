# One unsolved concern with this code is the 4096 spatial points subset considered
# for the field reconstruction as the entire domain of particles cannot be inputed for each time step in the dataset.

from __future__ import annotations
import json
import gc
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import h5py
import numpy as np

PREPROC_MAX_RSS_GB = float(os.environ.get("PREPROC_MAX_RSS_GB", "0"))
ROLLOUT_MEMORY_CAP_GB = float(os.environ.get("ROLLOUT_MEMORY_CAP_GB", "2.0"))


def _to_s32(values) -> np.ndarray:
    """Encode particle identities into fixed-width UTF-8 byte IDs."""
    arr = np.asarray(values).reshape(-1)
    if arr.dtype.kind in ("O", "U"):
        return np.char.encode(arr.astype(str), "utf-8").astype("S32")
    if arr.dtype.kind == "S":
        return arr.astype("S32")
    return arr.astype("S32")


def _rss_gb() -> float:
    """Return this process's resident memory use in GB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _check_rss_limit(case: str) -> None:
    if PREPROC_MAX_RSS_GB > 0 and _rss_gb() > PREPROC_MAX_RSS_GB:
        raise MemoryError(f"Preprocessing RSS exceeded PREPROC_MAX_RSS_GB after case={case}")


try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    cKDTree = None
    SCIPY_AVAILABLE = False
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except Exception:
    pv = None
    PYVISTA_AVAILABLE = False

# Particle and field arrays are read from HDF5; companion XMF files provide each frame's simulation-step time coordinate.
# The two roots below point to the Task 1 particle data and the Task 2 Eulerian field data that are combined later.

SCRIPT_DIR = Path(__file__).resolve().parents[2]

RAW_ROOT = Path("/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task1")
RAW_ROOT_CANDIDATES = [RAW_ROOT]
FIELD_ROOT = Path("/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task2")
FIELD_ROOT_CANDIDATES = [FIELD_ROOT]

# These settings determine which Task 1 case directories are discovered and how complete cases are assigned to the train, validation, and test sets.
# The split is case-based so that all frames and particle transitions from a given operating condition stay together.

DATASET_IDS: List[str] = []
AUTO_DISCOVER_TASK1_CASES = True
AUTO_ASSIGN_SPLITS_FROM_CASE_NAMES = True

TASK1_CASE_RE = re.compile(r"^(?P<aoa>\d+(?:\.\d+)?)deg_static_airfoil_(?P<speed>\d+(?:\.\d+)?)u_(?P<particles>\d+)p$")

VAL_AOA_DEGREES = [18, 24]
TRAIN_AOA_DEGREES = [aoa for aoa in range(10, 31, 2) if aoa not in VAL_AOA_DEGREES]
TEST_AOA_DEGREES = [11, 15, 21, 25, 27, 32]
IGNORE_AOA_DEGREES = [19]
TEST_NORMAL_AOA_DEGREES = [27]
TEST_SUPER_RESOLUTION_AOA_DEGREES: List[int] = []
TEST_UNSEEN_AOA_DEGREES = [32]
DEFAULT_PARTICLES_PER_STEP = 1
EXPECTED_PARTICLES_PER_STEP_BY_AOA: Dict[int, int] = {}

# These patterns identify the source files used by the preprocessing: dynamic particles, optional static particles, the VTK body geometry, and the Task 2 field data.
DYNAMIC_PARTICLE_H5_PATTERN = "static_airfoil_pfield.*.h5"
STATIC_PARTICLE_H5_PATTERN = "static_airfoil_staticpfield.*.h5"
VTK_PATTERN = "static_airfoil_Wing_vlm.*.vtk"
FIELD_H5_PATTERN = "static_airfoil_fdom.*.h5"
INCLUDE_STATIC_PARTICLES = True

# Intermediate merged frames are written first, followed by one consolidated dataset containing all particle and field-learning arrays.
OUT_ROOT = SCRIPT_DIR / "processed_data"
MERGED_ROOT = OUT_ROOT / "merged_frames"

# The configured seed is recorded here before the target mode is selected.
RANDOM_SEED = 42
TASK1_TARGET_MODE = "delta"

# Case metadata supplies physical quantities that are needed during preprocessing but are not stored as complete per-particle arrays.
CASE_METADATA_PATH = SCRIPT_DIR / "case_metadata.json"
STRICT_METADATA_VALIDATION = True

CASE_METADATA: Dict[str, Dict[str, Any]] = {}
DEFAULT_TASK1_DT = 0.0034

TRAIN_CASES: List[str] = []
VAL_CASES: List[str] = []
TEST_CASES: List[str] = []

# STATE_NAMES defines the seven quantities regarded as the particle state. TARGET_DELTA_NAMES defines the one-step quantities produced by comparing two consecutive frames, and FIELD_TARGET_NAMES defines the twelve Eulerian quantities used for Task 2.
# The source HDF5 files contain additional quantities such as circulation and volume, but those quantities are not part of the seven-dimensional propagated state in this script.
STATE_NAMES = ["x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma"]
TARGET_DELTA_NAMES = [
    "dx",
    "dy",
    "dz",
    "dGamma_x",
    "dGamma_y",
    "dGamma_z",
    "dsigma",
    "delta_u_x",
    "delta_u_y",
    "delta_u_z",
]

MAX_FIELD_QUERY_POINTS = 4096
FIELD_TARGET_NAMES = [
    "u_x", "u_y", "u_z",
    "dUx_dx", "dUx_dy", "dUx_dz",
    "dUy_dx", "dUy_dy", "dUy_dz",
    "dUz_dx", "dUz_dy", "dUz_dz",
]

ACTIVE_FIELD_QUERY_BOUNDS: Optional[Tuple[float, float, float, float, float, float]] = None
FIELD_STD_FLOOR = float(os.environ.get("FIELD_STD_FLOOR", "1e-6"))
FIELD_SUPERRESOLUTION_STRIDE = int(os.environ.get("FIELD_SUPERRESOLUTION_STRIDE", "5"))
FIELD_SUPERRESOLUTION_OFFSET = int(os.environ.get("FIELD_SUPERRESOLUTION_OFFSET", "4"))
XMF_TIME_MATCH_TOLERANCE = 1e-8

# These channels describe where a particle is relative to the body geometry. They are calculated from the VTK surface and are appended to the particle features when geometry conditioning is enabled.
USE_GEOMETRY_CHANNELS = True
GEOMETRY_NEAR_THRESHOLD = 0.02
GEOMETRY_CHANNEL_NAMES = [
    "geom_dist",
    "geom_nx",
    "geom_ny",
    "geom_nz",
    "geom_body_near",
]

# These channels describe the global operating condition and the particle frame within the stored sequence. Their values are repeated for every particle in the same frame.
USE_EXPLICIT_CONDITIONING = True
CONDITIONING_CHANNEL_NAMES = [
    "angle_of_attack",
    "freestream_x",
    "freestream_z",
    "phase",
]

# This list is the exact column ordering of the particle-model input matrix. The order matters because the numerical columns are saved together with these names and must be interpreted in the same order during training and inference.
PARTICLE_INPUT_FEATURES = [
    "x",
    "y",
    "z",
    "Gamma_x",
    "Gamma_y",
    "Gamma_z",
    "sigma",
    "u_x",
    "u_y",
    "u_z",
    "gradU_xx", "gradU_xy", "gradU_xz",
    "gradU_yx", "gradU_yy", "gradU_yz",
    "gradU_zx", "gradU_zy", "gradU_zz",
]
if USE_GEOMETRY_CHANNELS:
    PARTICLE_INPUT_FEATURES = PARTICLE_INPUT_FEATURES + GEOMETRY_CHANNEL_NAMES
if USE_EXPLICIT_CONDITIONING:
    PARTICLE_INPUT_FEATURES = PARTICLE_INPUT_FEATURES + CONDITIONING_CHANNEL_NAMES

# These settings control basic frame and geometry quality checks. The intent is to detect missing or collapsed information before the resulting arrays are used for training.
# MIN_PARTICLES_PER_FRAME = 64
EXPORT_FILTERED_FRAME_MANIFEST = True

STRICT_GEOMETRY_QA = True
GEOM_MIN_NONZERO_FRAC = 1e-4
GEOM_MIN_NEAR_FRAC = 1e-5

# This optional protocol is intended to reserve some frames from each training case for an internal validation set. It is separate from the case-level validation split defined above.
USE_DUAL_SPLIT_PROTOCOL = False
VAL_ID_FRACTION_FROM_TRAIN_CASES = 0.2
VAL_ID_MIN_FRAMES_PER_TRAIN_CASE = 8
VAL_ID_FRAME_STRIDE = 5
VAL_ID_FRAME_OFFSET = 2

CONDITIONING_ALLOWED_CONSTANT_CHANNELS: set = set()

FRAME_RE = re.compile(r"(\d+)(?!.*\d)")


# The functions in this section are small utilities used by the later data-loading and dataset-construction routines.
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def frame_id(path: Path) -> str:
    m = FRAME_RE.search(path.stem)
    return m.group(1).zfill(6) if m else path.stem


def read_xmf_time(path: Path) -> float:
    """Read the single simulation-step coordinate declared by an XMF file."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot parse XMF time from {path}: {exc}") from exc
    values = [element.get("Value") for element in root.iter()
              if element.tag.rsplit("}", 1)[-1] == "Time"]
    if len(values) != 1 or values[0] is None:
        raise ValueError(f"Expected exactly one <Time Value=...> in {path}, found {len(values)}")
    try:
        value = float(values[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric XMF time {values[0]!r} in {path}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite XMF time {value!r} in {path}")
    return value


def validate_task2_xmf_layout(xmf_path: Path, h5_path: Path) -> None:
    """Check that XDMF declares node coordinates and nodal U on the same 65^3 mesh."""
    try:
        root = ET.parse(xmf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot parse Task-2 XMF {xmf_path}: {exc}") from exc
    local = lambda element: element.tag.rsplit("}", 1)[-1]
    geometry = next((element for element in root.iter() if local(element) == "Geometry"), None)
    nodes_item = None if geometry is None else next(
        (child for child in geometry if local(child) == "DataItem"), None
    )
    u_attribute = next((element for element in root.iter()
                        if local(element) == "Attribute" and element.get("Name") == "U"), None)
    u_item = None if u_attribute is None else next(
        (child for child in u_attribute if local(child) == "DataItem"), None
    )
    if nodes_item is None or u_item is None:
        raise ValueError(f"{xmf_path}: expected Geometry/nodes and Attribute U DataItems")

    def reference(item):
        text = (item.text or "").strip()
        parts = text.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid HDF5 DataItem reference {text!r} in {xmf_path}")
        return Path(parts[0]).name, parts[1], tuple(int(v) for v in item.get("Dimensions", "").split())

    node_file, node_key, node_dims = reference(nodes_item)
    u_file, u_key, u_dims = reference(u_item)
    expected = 65 ** 3
    node_count = int(np.prod(node_dims[:-1])) if len(node_dims) == 2 and node_dims[-1] == 3 else (
        int(np.prod(node_dims[1:])) if len(node_dims) == 2 and node_dims[0] == 3 else -1
    )
    u_count = int(np.prod(u_dims[:-1])) if len(u_dims) == 4 and u_dims[-1] == 3 else (
        int(np.prod(u_dims[:-1])) if len(u_dims) == 2 and u_dims[-1] == 3 else -1
    )
    if (node_file != h5_path.name or u_file != h5_path.name or node_key != "nodes" or u_key != "U"
            or node_count != expected or u_count != expected
            or u_attribute.get("Center", "").lower() != "node"):
        raise ValueError(
            f"{xmf_path}: expected nodes and nodal U to reference {h5_path.name} with "
            f"{expected} entries; got nodes=({node_file}:{node_key}, {node_dims}), "
            f"U=({u_file}:{u_key}, {u_dims}, Center={u_attribute.get('Center')!r})"
        )


def match_task2_xmf_time(field_entries: List[Tuple[float, Path]], task1_time: float):
    matches = [(time, path) for time, path in field_entries
               if np.isclose(time, task1_time, rtol=0.0, atol=XMF_TIME_MATCH_TOLERANCE)]
    if len(matches) > 1:
        raise ValueError(f"Multiple Task-2 fields match Task-1 XMF time {task1_time}: {matches}")
    return matches[0] if matches else None

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
        return a.reshape(-1, 3).astype(np.float64)
    if a.ndim == 4 and a.shape[0] == 3:
        return np.moveaxis(a, 0, -1).reshape(-1, 3).astype(np.float64)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.astype(np.float64)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T.astype(np.float64)
    raise ValueError(f"Cannot parse vector field from shape={a.shape}")


# The HDF5 helpers read only the datasets requested by the preprocessing configuration. They fail immediately when a required dataset is absent instead of silently creating a substitute.
def read_h5_selected(path: Path, key_map: Dict[str, str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for out_key, in_key in key_map.items():
            if in_key not in f:
                raise KeyError(f"{path.name}: missing key {in_key}")
            out[out_key] = np.asarray(f[in_key])
    return out


# Search through the HDF5 hierarchy because particle IDs may be stored under a nested dataset name rather than at the file root.
def _read_particle_ids(path: Path, count: int, source: str) -> np.ndarray:
    candidates = {"particle_id", "particle_ids"}
    found = None
    with h5py.File(path, "r") as handle:
        def visit(name, node):
            nonlocal found
            if found is None and isinstance(node, h5py.Dataset) and name.rsplit("/", 1)[-1].lower() in candidates:
                value = np.asarray(node).reshape(-1)
                if len(value) == count:
                    found = value
        handle.visititems(visit)
    if found is None:
        return _to_s32([f"{source}:row:{i}" for i in range(count)])
    return _to_s32([f"{source}:id:{value}" for value in found])


def match_particle_identities(ids_t: np.ndarray, ids_tp1: np.ndarray):
    """Return the row indices that correspond to the same particle in the two frames and return the common persistent identities in that same order."""
    ids_t = np.asarray(ids_t).reshape(-1)
    ids_tp1 = np.asarray(ids_tp1).reshape(-1)
    if len(np.unique(ids_t)) != len(ids_t) or len(np.unique(ids_tp1)) != len(ids_tp1):
        raise ValueError("Particle identity arrays must be unique within each frame")
    ids_t_text = np.char.decode(ids_t, "utf-8") if ids_t.dtype.kind == "S" else ids_t
    ids_tp1_text = np.char.decode(ids_tp1, "utf-8") if ids_tp1.dtype.kind == "S" else ids_tp1
    next_lookup = {identity: j for j, identity in enumerate(ids_tp1_text)}
    indices_t = np.asarray([j for j, identity in enumerate(ids_t_text) if identity in next_lookup], dtype=np.int64)
    indices_tp1 = np.asarray([next_lookup[ids_t_text[j]] for j in indices_t], dtype=np.int64)
    identities = ids_t[indices_t]
    return indices_t, indices_tp1, identities


def _assign_advected_position_tracks(frames: List[Dict[str, object]], case: str) -> None:
    """Create conservative particle tracks from positions and velocities when the source files do not contain persistent particle IDs."""
    if not frames:
        return

    # Collect the static-particle coordinates across the sequence because static particles can be identified by stable spatial locations instead of advected motion.
    static_positions = []
    for frame in frames:
        xyz = np.stack([frame["state"][name] for name in ("x", "y", "z")], axis=1)
        static_positions.append(xyz[np.asarray(frame["static_mask"], dtype=bool)])
    nonempty_static_positions = [points for points in static_positions if len(points)]
    reference_static = nonempty_static_positions[0] if nonempty_static_positions else np.zeros((0, 3))
    stable_static_rows = bool(nonempty_static_positions) and all(
        points.shape == reference_static.shape and np.allclose(points, reference_static, rtol=0.0, atol=1e-10)
        for points in nonempty_static_positions[1:]
    )

    next_track_id = 0
    for frame in frames:
        ids = _to_s32(frame["particle_ids"]).copy()
        static_mask = np.asarray(frame["static_mask"], dtype=bool)
        xyz = np.stack([frame["state"][name] for name in ("x", "y", "z")], axis=1)
        decoded_ids = np.char.decode(ids, "utf-8")
        native_ids = np.char.find(decoded_ids, ":id:") >= 0
        fallback_static = static_mask & ~native_ids
        static_row_indices = np.flatnonzero(static_mask)
        for index in np.flatnonzero(fallback_static):
            if stable_static_rows:
                static_row = int(np.searchsorted(static_row_indices, index))
                ids[index] = f"s:{static_row}"
            else:
                x, y, z = np.round(xyz[index], decimals=9)
                digest = hashlib.blake2b(np.asarray([x, y, z], dtype="<f8").tobytes(), digest_size=12).hexdigest()
                ids[index] = f"sx:{digest}"
        # Dynamic particles without native IDs are marked for the position-based tracking pass that follows.
        fallback_dynamic = ~static_mask & ~native_ids
        frame["particle_ids"] = ids
        frame["advected_tracking_mask"] = fallback_dynamic
        frame["advected_match_fraction"] = 1.0 if not fallback_dynamic.any() else 0.0
        for index in np.flatnonzero(fallback_dynamic):
            ids[index] = f"d:{next_track_id}"
            next_track_id += 1

    for frame_index, (previous, current) in enumerate(zip(frames, frames[1:]), start=1):
        previous_ids = _to_s32(previous["particle_ids"])
        current_ids = _to_s32(current["particle_ids"]).copy()
        previous_indices = np.flatnonzero(previous["advected_tracking_mask"])
        current_indices = np.flatnonzero(current["advected_tracking_mask"])
        previous_xyz = np.stack([previous["state"][name] for name in ("x", "y", "z")], axis=1)
        current_xyz = np.stack([current["state"][name] for name in ("x", "y", "z")], axis=1)
        matched = np.zeros(len(current_indices), dtype=bool)

        if len(previous_indices) and len(current_indices) and SCIPY_AVAILABLE:
            step_dt = float(current.get("physical_time", 0.0)) - float(previous.get("physical_time", 0.0))
            if not np.isfinite(step_dt) or step_dt <= 0.0:
                raise ValueError(
                    f"Non-increasing physical XMF time between frames "
                    f"{previous.get('frame_id')} and {current.get('frame_id')}"
                )
            source_xyz = previous_xyz[previous_indices]
            target_xyz = current_xyz[current_indices]
            source_velocity = np.asarray(previous["velocity"], dtype=np.float64)[previous_indices]
            target_velocity = np.asarray(current["velocity"], dtype=np.float64)[current_indices]
            target_tree = cKDTree(target_xyz)
            forward_distance, forward_index = target_tree.query(source_xyz + source_velocity * step_dt, k=1)
            _, backward_index = cKDTree(source_xyz).query(target_xyz - target_velocity * step_dt, k=1)

            # Estimate a displacement-aware residual tolerance from target spacing.
            if len(target_xyz) > 1:
                spacing = cKDTree(target_xyz).query(target_xyz, k=2)[0][:, 1]
                spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
                base_tol = max(0.02, 3.0 * float(np.median(spacing))) if spacing.size else 0.02
            else:
                base_tol = 0.02
            source_local = np.arange(len(source_xyz))
            forward_residual = np.linalg.norm(
                target_xyz[forward_index] - (source_xyz + source_velocity * step_dt), axis=1)
            residual_limit = np.maximum(base_tol, 0.5 * np.linalg.norm(source_velocity, axis=1) * step_dt)
            valid = ((backward_index[forward_index] == source_local)
                     & (forward_distance <= base_tol)
                     & (forward_residual <= residual_limit))
            matched[forward_index[valid]] = True
            for src_local, dst_local in zip(source_local[valid], forward_index[valid]):
                current_ids[current_indices[dst_local]] = previous_ids[previous_indices[src_local]]

            # Globally resolve ambiguous leftovers using predicted position and velocity agreement.
            unmatched_sources = source_local[~np.isin(source_local, source_local[valid])]
            unmatched_targets = np.flatnonzero(~matched)
            if len(unmatched_sources) and len(unmatched_targets):
                from scipy.optimize import linear_sum_assignment

                predicted = source_xyz[unmatched_sources] + source_velocity[unmatched_sources] * step_dt
                position_cost = np.linalg.norm(predicted[:, None, :] - target_xyz[unmatched_targets][None, :, :], axis=2)
                velocity_cost = step_dt * np.linalg.norm(
                    source_velocity[unmatched_sources, None, :] - target_velocity[unmatched_targets][None, :, :], axis=2)
                cost = position_cost + velocity_cost
                source_assignment, target_assignment = linear_sum_assignment(cost)
                max_new_residual = max(2.0 * base_tol, 1.0)
                for src_offset, dst_offset in zip(source_assignment, target_assignment):
                    src_local = unmatched_sources[src_offset]
                    dst_local = unmatched_targets[dst_offset]
                    if cost[src_offset, dst_offset] > max_new_residual:
                        continue
                    matched[dst_local] = True
                    current_ids[current_indices[dst_local]] = previous_ids[previous_indices[src_local]]

        matched_count = int(matched.sum())
        new_count = int(len(current_indices) - matched_count)
        if not SCIPY_AVAILABLE:
            current["advected_match_fraction"] = 0.0 if len(current_indices) else 1.0
        else:
            current["advected_match_fraction"] = float(matched.mean()) if len(matched) else 1.0

        print(f"[track] case={case} frame={current.get('frame_id', frame_index)} "
              f"matched={matched_count}/{len(current_indices)} new={new_count}")
        current["particle_ids"] = _to_s32(current_ids)

    first_indices = np.flatnonzero(np.asarray(frames[0]["advected_tracking_mask"], dtype=bool))
    print(f"[track] case={case} frame={frames[0].get('frame_id', 0)} "
          f"matched=0/{len(first_indices)} new={len(first_indices)}")


# Task 2 provides an Eulerian field on a structured grid. Its default query domain is inferred from training-case meshes; FIELD_QUERY_BOUNDS can explicitly override it.
def _field_query_bounds() -> Optional[Tuple[float, float, float, float, float, float]]:
    raw = os.environ.get("FIELD_QUERY_BOUNDS", "").strip()
    if raw.lower() in {"none", "full", "all", "off", "false", "0"}:
        return None
    if not raw:
        return ACTIVE_FIELD_QUERY_BOUNDS
    parts = [float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    if len(parts) != 6:
        raise ValueError(
            "FIELD_QUERY_BOUNDS must have six values: xmin,xmax,ymin,ymax,zmin,zmax "
            f"or be 'none'; got {raw!r}"
        )
    xmin, xmax, ymin, ymax, zmin, zmax = parts
    if not (xmin < xmax and ymin < ymax and zmin < zmax):
        raise ValueError(f"Invalid FIELD_QUERY_BOUNDS ordering: {parts}")
    return xmin, xmax, ymin, ymax, zmin, zmax

def _filter_field_queries(coords: np.ndarray, values: np.ndarray, path: Path) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(values).all(axis=1)
    bounds = _field_query_bounds()
    keep = finite.copy()
    # Apply the configured spatial bounds only after rejecting non-finite coordinate or field values.
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        in_box = (
            (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax)
            & (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
            & (coords[:, 2] >= zmin) & (coords[:, 2] <= zmax)
        )
        keep &= in_box
    if not np.any(keep):
        raise ValueError(
            f"{path.name}: no finite field samples lie inside the training-derived query domain {bounds}"
        )
    return coords[keep].astype(np.float32), values[keep].astype(np.float32)


# This routine converts the Task 2 HDF5 arrays into a common point-wise representation, derives the requested field quantities, and applies the configured spatial and finite-value filtering.
def read_field_grid_h5(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        coords = as_xyz(np.asarray(f["nodes"]))
        raw_velocity = np.asarray(f["U"])
        if raw_velocity.ndim == 4 and raw_velocity.shape[-1] == 3:
            _validate_structured_node_order(coords, raw_velocity.shape[:3], path)
        elif raw_velocity.ndim == 4 and raw_velocity.shape[0] == 3:
            _validate_structured_node_order(coords, raw_velocity.shape[1:], path)
        vel = as_vec_field(raw_velocity)

    expected = 65 ** 3
    if len(coords) != expected or len(vel) != expected:
        raise ValueError(
            f"{path}: expected exactly {expected} Task-2 nodes and velocity rows, "
            f"got N_nodes={len(coords)}, N_U={len(vel)}"
        )
    if vel.shape != coords.shape:
        raise ValueError(f"{path}: node/velocity shapes disagree: {coords.shape} vs {vel.shape}")

    from .hdf5 import physical_field_values
    invalid_velocity = ~np.isfinite(vel).all(axis=1)
    derivative_velocity = vel
    if invalid_velocity.any():
        axes = tuple(np.unique(coords[:, axis]) for axis in range(3))
        indices = tuple(np.searchsorted(axes[axis], coords[:, axis]) for axis in range(3))
        velocity_grid = np.full(tuple(len(axis) for axis in axes) + (3,), np.nan, dtype=np.float32)
        velocity_grid[indices] = vel
        invalid_grid = ~np.isfinite(velocity_grid).all(axis=-1)
        try:
            from scipy.ndimage import distance_transform_edt

            nearest = distance_transform_edt(invalid_grid, return_distances=False, return_indices=True)
            velocity_grid[invalid_grid] = velocity_grid[tuple(nearest[:, invalid_grid])]
        except ImportError:
            velocity_grid[invalid_grid] = 0.0
        derivative_velocity = velocity_grid[indices]
    combined = physical_field_values(coords, derivative_velocity)
    if invalid_velocity.any():
        combined[invalid_velocity] = np.nan

    return _filter_field_queries(coords, combined, path)


def _validate_structured_node_order(coords: np.ndarray, spatial_shape: Tuple[int, ...], path: Path) -> None:
    """Ensure flattened nodes follow the tensor-product order of a 3-D HDF5 U array."""
    if len(spatial_shape) != 3 or int(np.prod(spatial_shape)) != len(coords):
        raise ValueError(f"{path}: U spatial shape {spatial_shape} does not match {len(coords)} nodes")
    grid = np.asarray(coords).reshape((*spatial_shape, 3))
    varying_dimensions = []
    for component in range(3):
        matching_dimension = None
        for dimension in range(3):
            selector = [0, 0, 0]
            selector[dimension] = slice(None)
            line = grid[tuple(selector) + (component,)]
            if np.ptp(line) <= 0.0:
                continue
            line_shape = [1, 1, 1]
            line_shape[dimension] = len(line)
            expected = np.broadcast_to(line.reshape(line_shape), spatial_shape)
            if np.allclose(grid[..., component], expected, rtol=1e-7, atol=1e-9):
                matching_dimension = dimension
                break
        if matching_dimension is None:
            raise ValueError(
                f"{path}: node order does not reshape to a rectilinear grid matching U's "
                f"tensor dimensions {spatial_shape}; coordinate component {component} is not separable"
            )
        varying_dimensions.append(matching_dimension)
    if len(set(varying_dimensions)) != 3:
        raise ValueError(
            f"{path}: XMF node ordering does not map one-to-one to U's three spatial axes; "
            f"coordinate axes map to {varying_dimensions}"
        )


def _infer_training_field_bounds(
    field_files_by_case: Dict[str, List[Tuple[float, Path]]],
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Infer a shared query domain from training-case Task-2 meshes only."""
    raw_override = os.environ.get("FIELD_QUERY_BOUNDS", "").strip()
    if raw_override:
        return _field_query_bounds()
    lows, highs = [], []
    for case in TRAIN_CASES:
        case_fields = field_files_by_case.get(str(case), [])
        for _, path in case_fields[:1]:
            with h5py.File(path, "r") as handle:
                coords = as_xyz(np.asarray(handle["nodes"]))
            coords = coords[np.isfinite(coords).all(axis=1)]
            if coords.size:
                lows.append(coords.min(axis=0))
                highs.append(coords.max(axis=0))
    if not lows:
        return None
    lower, upper = np.min(lows, axis=0), np.max(highs, axis=0)
    if not np.all(upper > lower):
        raise ValueError(f"Inferred training Task-2 domain has degenerate bounds: {lower}, {upper}")
    return (float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1]),
            float(lower[2]), float(upper[2]))


# These helpers deal with small shape and indexing issues that occur repeatedly when the frame-level arrays are converted into the final flattened dataset.
def _require_scalar(arr: np.ndarray, n: int, name: str) -> np.ndarray:
    values = np.asarray(arr).reshape(-1)
    if values.size != n:
        raise ValueError(f"{name} must contain exactly {n} values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values.astype(np.float64)

def _rows_from_pair_ids(pair_ranges: List[Tuple], pair_ids: np.ndarray) -> np.ndarray:
    chunks = []
    for i in pair_ids:
        _, _, _, s, e, _ = pair_ranges[int(i)]
        chunks.append(np.arange(int(s), int(e), dtype=np.int64))
    return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.int64)


# Every discovered case must belong to exactly one declared split. This prevents a case from being silently omitted or accidentally appearing in more than one split.
def _validate_case_split(all_cases: List[str]) -> None:
    if len(all_cases) == 0:
        raise ValueError(
            "No cases discovered from merged frames. This usually means raw-data path/pattern mismatch. "
            "Check RAW_ROOT and INPUT/OUTPUT patterns."
        )
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

ACTIVE_CASE_METADATA: Dict[str, Dict[str, Any]] = {}
ACTIVE_CONDITIONING_CHANNEL_NAMES: List[str] = []


# The metadata file allows physical case information to be supplied explicitly and to override values inferred from the case name.
def _load_case_metadata_overrides() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    # A missing metadata file is allowed because the case name can provide the default AoA and free-stream magnitude.
    if not CASE_METADATA_PATH.exists():
        return {}, {}
    payload = json.loads(CASE_METADATA_PATH.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"CASE_METADATA_PATH must contain a JSON object: {CASE_METADATA_PATH}")
    defaults = {}
    for key in ["_defaults", "defaults", "__defaults__"]:
        if key in payload and isinstance(payload[key], dict):
            # Use the first supported defaults block found in the metadata JSON as the common starting point for case-specific metadata.
            defaults = dict(payload[key])
            break
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in payload.items():
        if str(k) in {"_defaults", "defaults", "__defaults__"}:
            continue
        out[str(k)] = dict(v) if isinstance(v, dict) else {}
    return out, defaults


# The source directories may be overridden through environment variables. When no override is supplied, the fixed paths defined above are used.
def _resolve_raw_root() -> Path:
    env_raw = os.environ.get("RAW_ROOT", "").strip()
    if env_raw:
        p = Path(env_raw)
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(
            f"RAW_ROOT is set but invalid: {p}. "
            "Fix the env var or unset it."
        )

    for cand in RAW_ROOT_CANDIDATES:
        c = Path(cand)
        if c.exists() and c.is_dir():
            return c
    tried = ", ".join([str(Path(c)) for c in RAW_ROOT_CANDIDATES])
    raise FileNotFoundError(
        "Could not locate Task-1 dataset root. Tried: "
        f"{tried}. Update RAW_ROOT or mount the external drive."
    )

def _resolve_field_root() -> Path:
    env_raw = os.environ.get("FIELD_ROOT", os.environ.get("TASK2_ROOT", "")).strip()
    if env_raw:
        p = Path(env_raw)
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(
            f"FIELD_ROOT/TASK2_ROOT is set but invalid: {p}. "
            "Fix the env var or unset it."
        )

    for cand in FIELD_ROOT_CANDIDATES:
        c = Path(cand)
        if c.exists() and c.is_dir():
            return c
    tried = ", ".join([str(Path(c)) for c in FIELD_ROOT_CANDIDATES])
    raise FileNotFoundError(
        "Could not locate Task-2 field-grid root. Tried: "
        f"{tried}. Update FIELD_ROOT/TASK2_ROOT or mount the external drive."
    )


# The case name encodes the angle of attack, free-stream speed, and particle resolution. These values are used both for case discovery and for constructing default physical metadata.
def _parse_task1_case_name(case: str) -> Optional[Dict[str, float]]:
    m = TASK1_CASE_RE.match(str(case))
    if not m:
        return None
    return {
        "aoa_deg": float(m.group("aoa")),
        "magVinf": float(m.group("speed")),
        "particles_per_step": float(m.group("particles")),
    }

def _expected_particles_per_step_for_aoa(aoa_deg: float) -> int:
    aoa_key = int(round(float(aoa_deg)))
    return int(EXPECTED_PARTICLES_PER_STEP_BY_AOA.get(aoa_key, DEFAULT_PARTICLES_PER_STEP))

def _case_has_expected_particle_count(case: str) -> Tuple[bool, str]:
    parsed = _parse_task1_case_name(case)
    if parsed is None:
        return False, "case name does not match Task-1 naming convention"
    observed = int(round(float(parsed["particles_per_step"])))
    expected = _expected_particles_per_step_for_aoa(float(parsed["aoa_deg"]))
    if observed != expected:
        return (
            False,
            f"expected {expected}p for AoA {parsed['aoa_deg']:g}, found {observed}p",
        )
    return True, "ok"


# Automatic discovery keeps only directories that have the expected case-name format, the expected particle resolution, and at least one dynamic particle file.
def _discover_task1_case_ids(root: Path) -> List[str]:
    if not AUTO_DISCOVER_TASK1_CASES:
        return [str(x) for x in DATASET_IDS if str(x).strip()]

    out: List[str] = []
    skipped_resolution: List[Tuple[str, str]] = []
    skipped_ignored_aoa: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        parsed = _parse_task1_case_name(p.name)
        if parsed is None:
            continue
        aoa = int(round(float(parsed["aoa_deg"])))
        if aoa in IGNORE_AOA_DEGREES:
            skipped_ignored_aoa.append(p.name)
            continue
        ok_resolution, why_resolution = _case_has_expected_particle_count(p.name)
        if not ok_resolution:
            skipped_resolution.append((p.name, why_resolution))
            continue
        if len(list(p.glob(DYNAMIC_PARTICLE_H5_PATTERN))) == 0:
            continue
        out.append(p.name)
    if skipped_resolution:
        print("[data] skipped case folders with unintended particle resolution:")
        for name, why in skipped_resolution:
            print(f"  - {name}: {why}")
    if skipped_ignored_aoa:
        print("[data] skipped ignored AoA case folders:")
        for name in skipped_ignored_aoa:
            print(f"  - {name}")
    if not out:
        raise RuntimeError(
            f"No Task-1 case folders found in {root}. "
            f"Expected names like 10deg_static_airfoil_10u_1p and files matching {DYNAMIC_PARTICLE_H5_PATTERN}."
        )
    return out


# Each case is assigned to a split using its angle of attack. Because the assignment is made at case level, individual frames from one case cannot be distributed across different splits.
def _assign_case_splits_from_names(case_ids: List[str]) -> None:
    if not AUTO_ASSIGN_SPLITS_FROM_CASE_NAMES:
        return

    global TRAIN_CASES, VAL_CASES, TEST_CASES
    train, val, test, unassigned = [], [], [], []
    for case in sorted(case_ids):
        parsed = _parse_task1_case_name(case)
        if parsed is None:
            unassigned.append(case)
            continue
        aoa = int(round(float(parsed["aoa_deg"])))
        if aoa in IGNORE_AOA_DEGREES:
            continue
        if aoa in TRAIN_AOA_DEGREES:
            train.append(case)
        elif aoa in VAL_AOA_DEGREES:
            val.append(case)
        elif aoa in TEST_AOA_DEGREES:
            test.append(case)
        else:
            unassigned.append(case)

    if unassigned:
        raise ValueError(
            "Some discovered cases do not match TRAIN_AOA_DEGREES / VAL_AOA_DEGREES / TEST_AOA_DEGREES: "
            + ", ".join(unassigned)
        )

    TRAIN_CASES = train
    VAL_CASES = val
    TEST_CASES = test


# This function gives the test cases descriptive labels for later evaluation. It distinguishes an angle outside the training range from an angle that lies within the training range.
def _test_case_role_from_metadata(meta: Dict[str, Any]) -> str:
    aoa = int(round(float(meta["aoa_deg"])))
    if aoa in TEST_UNSEEN_AOA_DEGREES:
        return "testing_unseen_angle"
    if aoa in TEST_NORMAL_AOA_DEGREES:
        return "testing_normal"
    return "testing_other"


# This routine converts the available metadata into the exact physical quantities required by the preprocessing, particularly the three-component free-stream velocity and the time step between recorded frames.
def _resolve_case_metadata(meta: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    if "aoa_deg" not in meta or meta["aoa_deg"] is None:
        return None, "missing aoa_deg"
    try:
        aoa_deg = float(meta["aoa_deg"])
    except Exception:
        return None, "aoa_deg not numeric"
    if not np.isfinite(aoa_deg):
        return None, "aoa_deg non-finite"

    fs = meta.get("freestream", None)
    mag_vinf = meta.get("magVinf", None)
    if fs is None:
        if mag_vinf is None:
            return None, "missing freestream and magVinf"
        try:
            mag_vinf = float(mag_vinf)
        except Exception:
            return None, "magVinf not numeric"
        if not np.isfinite(mag_vinf) or mag_vinf <= 0.0:
            return None, "magVinf must be finite and >0"
        rad = np.deg2rad(aoa_deg)
        fs_arr = np.asarray(
            [mag_vinf * np.cos(rad), 0.0, mag_vinf * np.sin(rad)],
            dtype=np.float64,
        )
    else:
        try:
            fs_arr = np.asarray(fs, dtype=np.float64).reshape(-1)
        except Exception:
            return None, "freestream not numeric"
        if fs_arr.shape[0] != 3:
            return None, "freestream must have 3 components"
        if not np.all(np.isfinite(fs_arr)):
            return None, "freestream has non-finite values"
        if mag_vinf is None:
            mag_vinf = float(np.linalg.norm(fs_arr))
        else:
            try:
                mag_vinf = float(mag_vinf)
            except Exception:
                return None, "magVinf not numeric"

    # Use a directly supplied time step when available; otherwise derive it from the configured duration information below.
    dt_raw = meta.get("dt", None)
    if dt_raw is not None:
        try:
            dt = float(dt_raw)
        except Exception:
            return None, "dt not numeric"
        if not np.isfinite(dt) or dt <= 0.0:
            return None, "dt must be finite and >0"
    else:
        nsteps = meta.get("nsteps", None)
        if nsteps is None:
            return None, "missing dt and nsteps"
        try:
            nsteps = int(nsteps)
        except Exception:
            return None, "nsteps not integer"
        if nsteps <= 0:
            return None, "nsteps must be >0"

        # When dt is not given directly, derive it from the declared number of simulation steps and the available total-duration definition.
        if "ttot_seconds" in meta and meta["ttot_seconds"] is not None:
            try:
                ttot_seconds = float(meta["ttot_seconds"])
            except Exception:
                return None, "ttot_seconds not numeric"
            if not np.isfinite(ttot_seconds) or ttot_seconds <= 0.0:
                return None, "ttot_seconds must be finite and >0"
            dt = ttot_seconds / float(nsteps)
        elif "ttot_constant" in meta and meta["ttot_constant"] is not None:
            try:
                ttot_constant = float(meta["ttot_constant"])
            except Exception:
                return None, "ttot_constant not numeric"
            if not np.isfinite(ttot_constant) or ttot_constant <= 0.0:
                return None, "ttot_constant must be finite and >0"
            if mag_vinf is None:
                return None, "magVinf required for ttot_constant formula"
            if not np.isfinite(float(mag_vinf)) or float(mag_vinf) <= 0.0:
                return None, "magVinf must be finite and >0 for ttot_constant formula"
            ttot_seconds = float(ttot_constant) / float(mag_vinf)
            dt = ttot_seconds / float(nsteps)
        else:
            return None, "missing dt and no supported dt formula keys"

    resolved = {
        "aoa_deg": float(aoa_deg),
        "freestream": fs_arr.astype(np.float64).reshape(3),
        "dt": float(dt),
        "magVinf": float(np.linalg.norm(fs_arr)),
    }
    return resolved, "ok"


# Metadata resolution combines defaults, case-specific overrides, and information extracted from the case name. Explicit metadata takes precedence over the inferred values.
def _prepare_case_metadata(all_cases: List[str]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in CASE_METADATA.items()}
    overrides, defaults = _load_case_metadata_overrides()
    merged.update(overrides)

    for c in all_cases:
        if str(c) not in merged:
            parsed = _parse_task1_case_name(str(c))
            if parsed is None:
                merged[str(c)] = {}
            else:
                merged[str(c)] = {
                    "aoa_deg": float(parsed["aoa_deg"]),
                    "magVinf": float(parsed["magVinf"]),
                    "dt": DEFAULT_TASK1_DT,
                    "particles_per_step": int(parsed["particles_per_step"]),
                }

    # Resolve and validate the final metadata for every discovered case before any particle pairs are constructed.
    active: Dict[str, Dict[str, Any]] = {}
    missing: Dict[str, str] = {}
    for case in all_cases:
        meta = dict(defaults)
        meta.update(merged.get(case, {}) or {})
        resolved, why = _resolve_case_metadata(meta)
        if resolved is None:
            missing[str(case)] = why
            continue
        active[str(case)] = {
            "aoa_deg": float(resolved["aoa_deg"]),
            "freestream": np.asarray(resolved["freestream"], dtype=np.float64).reshape(3),
            "dt": float(resolved["dt"]),
            "magVinf": float(resolved["magVinf"]),
            "particles_per_step": int(round(float(meta.get("particles_per_step", 1)))),
        }

    # With strict validation enabled, incomplete metadata stops preprocessing when conditioning is requested rather than allowing an unspecified physical condition into the feature matrix.
    if USE_EXPLICIT_CONDITIONING and STRICT_METADATA_VALIDATION and missing:
        details = ", ".join([f"{k}({v})" for k, v in sorted(missing.items())])
        raise ValueError(
            "Conditioning is enabled but metadata is incomplete/invalid for cases: "
            f"{details}. Fill CASE_METADATA or provide {CASE_METADATA_PATH}."
        )
    if missing and not USE_EXPLICIT_CONDITIONING:
        print(f"[meta] conditioning disabled, allowing missing metadata for cases: {sorted(missing.keys())}")
    elif missing and not STRICT_METADATA_VALIDATION:
        print(f"[meta] warning: metadata missing for cases: {sorted(missing.keys())}")

    return active

def _case_meta(case: str) -> Dict[str, object]:
    meta = ACTIVE_CASE_METADATA.get(str(case), None)
    if meta is None:
        if USE_EXPLICIT_CONDITIONING and STRICT_METADATA_VALIDATION:
            raise ValueError(f"Missing validated metadata for case={case}")
        return {
            "aoa_deg": 0.0,
            "freestream": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "dt": 1.0,
            "particles_per_step": 1,
        }
    return {
        "aoa_deg": float(meta["aoa_deg"]),
        "freestream": np.asarray(meta["freestream"], dtype=np.float64).reshape(3),
        "dt": float(meta["dt"]),
        "particles_per_step": int(meta.get("particles_per_step", 1)),
    }


# The Task 1 particle data can be paired with a VTK representation of the airfoil or wing. These routines load the geometry, obtain surface normals when available, and prepare a nearest-neighbour search for particle-to-body queries.
_VTK_GEOM_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, object]] = {}
_GEOM_BACKEND_NOTICE_PRINTED = False

# Divide each vector by its magnitude while protecting against division by a value below eps.
def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def _point_normals_from_cells(points: np.ndarray, cells: List[np.ndarray]) -> np.ndarray:
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


# This is the fallback reader for legacy ASCII VTK files. It extracts the point coordinates and, when surface cell connectivity is available, estimates point normals from the connected cells.
def _load_legacy_vtk_ascii(vp: Path) -> Tuple[np.ndarray, np.ndarray]:
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


# Geometry is cached by VTK filename because the same surface is queried repeatedly across many particle frames. The cache stores the geometry, normals, and nearest-neighbour search tree.
def _load_vtk_geom(vtk_path: str):
    if not vtk_path:
        return None, None, None
    if vtk_path in _VTK_GEOM_CACHE:
        return _VTK_GEOM_CACHE[vtk_path]

    vp = Path(vtk_path)
    if not vp.exists():
        _VTK_GEOM_CACHE[vtk_path] = (None, None, None)
        return _VTK_GEOM_CACHE[vtk_path]

    try:
        # Prefer PyVista when it is available because it handles the VTK structure directly and can expose stored point normals.
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

            # If no usable point normals were found in the file, estimate normals from the surface cells instead.
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
                    pts2, nrm2 = _load_legacy_vtk_ascii(vp)
                    pts, nrm = pts2, nrm2
        else:
            pts, nrm = _load_legacy_vtk_ascii(vp)

        # Build the nearest-neighbour search structure once for this geometry so particle queries are efficient.
        tree = cKDTree(pts) if (SCIPY_AVAILABLE and pts is not None and pts.size > 0) else None
        _VTK_GEOM_CACHE[vtk_path] = (pts, nrm, tree)
    except Exception as e:
        global _GEOM_BACKEND_NOTICE_PRINTED
        if not _GEOM_BACKEND_NOTICE_PRINTED:
            print(f"[geom] warning: failed to load VTK geometry ({e}). Geometry channels will be zeros for affected frames.")
            _GEOM_BACKEND_NOTICE_PRINTED = True
        _VTK_GEOM_CACHE[vtk_path] = (None, None, None)

    return _VTK_GEOM_CACHE[vtk_path]


# For each particle, the nearest geometry point supplies a distance and an associated surface normal. The binary near-body indicator is obtained by comparing that distance with GEOMETRY_NEAR_THRESHOLD.
def _particle_geometry_features(xyz: np.ndarray, vtk_path: str, n: int) -> Dict[str, np.ndarray]:
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


# These routines summarize the geometry features and provide checks for the common failure mode in which the geometry file was not loaded and the resulting input channels are all zero.
def _print_geometry_channel_stats(X: np.ndarray, feature_names: List[str], tag: str) -> None:
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

def _geometry_frame_report(case: str, frame: str, n_particles: int, vtk_path: str, geom_feat: Dict[str, np.ndarray]) -> Dict[str, Any]:
    dist = np.asarray(geom_feat.get("geom_dist", np.zeros(n_particles)), dtype=np.float64)
    nx = np.asarray(geom_feat.get("geom_nx", np.zeros(n_particles)), dtype=np.float64)
    ny = np.asarray(geom_feat.get("geom_ny", np.zeros(n_particles)), dtype=np.float64)
    nz = np.asarray(geom_feat.get("geom_nz", np.zeros(n_particles)), dtype=np.float64)
    near = np.asarray(geom_feat.get("geom_body_near", np.zeros(n_particles)), dtype=np.float64)
    nrm_mag = np.sqrt(nx * nx + ny * ny + nz * nz)

    return {
        "case": str(case),
        "frame": str(frame),
        "n_particles": int(n_particles),
        "vtk_path": str(vtk_path),
        "geom_dist_mean": float(np.mean(dist)) if dist.size else 0.0,
        "geom_dist_p95": float(np.percentile(dist, 95)) if dist.size else 0.0,
        "geom_dist_nonzero_frac": float(np.mean(dist > 0.0)) if dist.size else 0.0,
        "geom_body_near_frac": float(np.mean(near > 0.5)) if near.size else 0.0,
        "normal_mag_mean": float(np.mean(nrm_mag)) if nrm_mag.size else 0.0,
        "normal_mag_var": float(np.var(nrm_mag)) if nrm_mag.size else 0.0,
        "normal_nx_var": float(np.var(nx)) if nx.size else 0.0,
        "normal_ny_var": float(np.var(ny)) if ny.size else 0.0,
        "normal_nz_var": float(np.var(nz)) if nz.size else 0.0,
    }

def _geometry_case_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_case[str(e["case"])].append(e)
    out: Dict[str, Any] = {}
    for case, arr in sorted(by_case.items()):
        dist_nonzero = np.asarray([a["geom_dist_nonzero_frac"] for a in arr], dtype=np.float64)
        near_frac = np.asarray([a["geom_body_near_frac"] for a in arr], dtype=np.float64)
        nx_var = np.asarray([a["normal_nx_var"] for a in arr], dtype=np.float64)
        ny_var = np.asarray([a["normal_ny_var"] for a in arr], dtype=np.float64)
        nz_var = np.asarray([a["normal_nz_var"] for a in arr], dtype=np.float64)
        out[case] = {
            "n_frames": int(len(arr)),
            "geom_dist_nonzero_frac_mean": float(np.mean(dist_nonzero)),
            "geom_body_near_frac_mean": float(np.mean(near_frac)),
            "normal_var_mean": float(np.mean(nx_var + ny_var + nz_var)),
        }
    return out

def _validate_geometry_report(entries: List[Dict[str, Any]]) -> None:
    if not USE_GEOMETRY_CHANNELS or len(entries) == 0:
        return
    case_sum = _geometry_case_summary(entries)
    bad_cases: List[str] = []
    for case, s in case_sum.items():
        if s["geom_dist_nonzero_frac_mean"] < GEOM_MIN_NONZERO_FRAC:
            bad_cases.append(f"{case}:geom_dist_nonzero={s['geom_dist_nonzero_frac_mean']:.3e}")
        if s["geom_body_near_frac_mean"] < GEOM_MIN_NEAR_FRAC:
            bad_cases.append(f"{case}:near_frac={s['geom_body_near_frac_mean']:.3e}")
    if bad_cases:
        msg = "Geometry channels appear collapsed for cases -> " + ", ".join(bad_cases)
        if STRICT_GEOMETRY_QA:
            raise RuntimeError(msg)
        print(f"[geom] warning: {msg}")


# This function contains the configured minimum-particle criterion for deciding whether a frame has enough data to be used.
# def _frame_is_usable(n_particles: int) -> Tuple[bool, str]:
#     if n_particles < int(MIN_PARTICLES_PER_FRAME):
#         return False, f"n_particles<{MIN_PARTICLES_PER_FRAME}"
#     return True, "ok"


# This routine optionally removes a regularly spaced subset of frames from the training cases for an internal validation check, while leaving the cases themselves in the training group.
def _train_id_val_id_split_by_case(frame_ranges: List[Tuple], train_frame_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not USE_DUAL_SPLIT_PROTOCOL or train_frame_ids.size == 0:
        return train_frame_ids, np.zeros((0,), dtype=np.int64)

    by_case: Dict[str, List[int]] = defaultdict(list)
    for i in train_frame_ids.tolist():
        case = str(frame_ranges[int(i)][0])
        by_case[case].append(int(i))

    train_out: List[int] = []
    val_id_out: List[int] = []
    for case, ids in sorted(by_case.items()):
        ids_sorted = sorted(ids, key=lambda j: int(frame_ranges[int(j)][1]))
        n = len(ids_sorted)
        if n <= 1:
            train_out.extend(ids_sorted)
            continue

        stride = max(int(VAL_ID_FRAME_STRIDE), 2)
        offset = min(max(int(VAL_ID_FRAME_OFFSET), 0), stride - 1)
        val_ids = ids_sorted[offset::stride]

        min_val = min(max(int(VAL_ID_MIN_FRAMES_PER_TRAIN_CASE), 1), max(n - 1, 1))
        if len(val_ids) < min_val:
            needed = min_val - len(val_ids)
            existing = set(val_ids)
            supplement = [j for j in ids_sorted if j not in existing][:needed]
            val_ids.extend(supplement)

        val_set = set(val_ids)
        train_ids = [j for j in ids_sorted if j not in val_set]
        if not train_ids:
            train_ids = ids_sorted[:-1]
            val_ids = ids_sorted[-1:]

        val_id_out.extend(val_ids)
        train_out.extend(train_ids)

    return np.asarray(sorted(train_out), dtype=np.int64), np.asarray(sorted(val_id_out), dtype=np.int64)


# These statistics describe how many frames and particles are present in a split and summarize the magnitudes of the velocity and velocity-gradient portions of the supplied array.
def _split_stats_from_rows(
    Y: np.ndarray,
    frame_ranges: List[Tuple],
    split_ids: np.ndarray,
    split_name: str,
) -> Dict[str, Any]:
    if split_ids.size == 0:
        return {"split": split_name, "n_frames": 0}
    rows = []
    case_counts: Dict[str, int] = defaultdict(int)
    n_particles = []
    for i in split_ids.tolist():
        case, _, s, e, n = frame_ranges[int(i)]
        s = int(s)
        e = int(e)
        rows.append(np.arange(s, e, dtype=np.int64))
        case_counts[str(case)] += 1
        n_particles.append(int(n))
    idx = np.concatenate(rows) if rows else np.zeros((0,), dtype=np.int64)
    yy = Y[idx]
    vel = np.linalg.norm(yy[:, :3], axis=1)
    grd = np.linalg.norm(yy[:, 3:], axis=1)

    def _q(v: np.ndarray, q: float) -> float:
        return float(np.quantile(v, q)) if v.size else float("nan")

    return {
        "split": split_name,
        "n_frames": int(split_ids.size),
        "case_counts": dict(case_counts),
        "n_particles": {
            "min": int(np.min(n_particles)),
            "median": float(np.median(n_particles)),
            "p90": float(np.quantile(n_particles, 0.9)),
            "max": int(np.max(n_particles)),
        },
        "u_mag": {
            "mean": float(np.mean(vel)),
            "p50": _q(vel, 0.5),
            "p95": _q(vel, 0.95),
            "max": float(np.max(vel)),
        },
        "grad_u_mag": {
            "mean": float(np.mean(grd)),
            "p50": _q(grd, 0.5),
            "p95": _q(grd, 0.95),
            "max": float(np.max(grd)),
        },
    }


# A conditioning channel should vary across the training data when it is intended to distinguish operating conditions. This check reports channels that are effectively constant on the training rows.
def _assert_conditioning_variance(X: np.ndarray, feature_names: List[str], train_rows: np.ndarray) -> None:
    if not USE_EXPLICIT_CONDITIONING:
        return
    dead = []
    informative = []
    for ch in ACTIVE_CONDITIONING_CHANNEL_NAMES:
        if ch not in feature_names:
            dead.append(f"{ch}(missing)")
            continue
        idx = feature_names.index(ch)
        v = float(np.var(X[train_rows, idx]))
        if v <= 1e-12:
            if ch in CONDITIONING_ALLOWED_CONSTANT_CHANNELS:
                print(f"[cond] info: {ch} is constant by design; allowing it.")
                continue
            dead.append(f"{ch}(var~0)")
        else:
            informative.append(ch)

    if dead and len(informative) == 0:
        raise RuntimeError(
            "Conditioning channels are non-informative on train rows (after allowed constants): "
            + ", ".join(dead)
            + ". Provide valid metadata or disable conditioning."
        )
    if dead and len(informative) > 0:
        print(
            "[cond] warning: some conditioning channels are constant, "
            f"but proceeding because informative channels exist: {informative}. "
            f"Constant channels: {dead}"
        )


# The source dynamic and static particle files use the same physical field names. When static particles are included, the corresponding arrays are concatenated so that later processing works with one combined particle set per frame.
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
def _merge_particle_payloads(dynamic_payload: Dict[str, np.ndarray], static_payload: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if static_payload is None or not INCLUDE_STATIC_PARTICLES:
        return {k: np.asarray(v) for k, v in dynamic_payload.items()}

    out: Dict[str, np.ndarray] = {}
    for key in dynamic_payload:
        a = np.asarray(dynamic_payload[key])
        b = np.asarray(static_payload[key])
        if a.ndim == 0 or b.ndim == 0:
            out[key] = a
        else:
            out[key] = np.concatenate([a, b], axis=0)
    return out


# Each Task 1 frame is converted into one intermediate NPZ file. This step puts dynamic and static particles into a common array and records the source paths needed by later geometry and field processing.
def merge_frames() -> List[Path]:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"RAW_ROOT does not exist: {RAW_ROOT}. "
            "Mount/check external drive or update RAW_ROOT."
        )

    ensure_dir(MERGED_ROOT)
    merged: List[Path] = []
    missing_dataset_dirs: List[str] = []

    for ds in DATASET_IDS:
        root = RAW_ROOT / ds
        if not root.exists():
            missing_dataset_dirs.append(str(root))
            continue

        dynamic_h5 = sorted(root.glob(DYNAMIC_PARTICLE_H5_PATTERN))
        static_h5 = sorted(root.glob(STATIC_PARTICLE_H5_PATTERN))
        particle_xmf = {frame_id(p): p for p in sorted(root.glob("static_airfoil_pfield.*.xmf"))}
        vtk = {frame_id(p): p for p in sorted(root.glob(VTK_PATTERN))}

        dynamic_map = {frame_id(p): p for p in dynamic_h5}
        static_map = {frame_id(p): p for p in static_h5}
        common = sorted(dynamic_map)
        print(
            f"[merge] {ds}: pfield={len(dynamic_h5)} "
            f"static_pfield={len(static_h5)} vtk={len(vtk)} paired={len(common)}"
        )

        out_dir = ensure_dir(MERGED_ROOT / ds)
        for fr in common:
            pin = dynamic_map[fr]
            pstatic = static_map.get(fr)
            dynamic_payload = read_h5_selected(pin, INPUT_KEYS)
            static_payload = read_h5_selected(pstatic, INPUT_KEYS) if pstatic is not None else None
            dynamic_ids = _read_particle_ids(pin, len(as_xyz(dynamic_payload["particle_xyz"])), "dynamic")
            static_ids = (_read_particle_ids(pstatic, len(as_xyz(static_payload["particle_xyz"])), "static")
                          if pstatic is not None and static_payload is not None else np.zeros(0, dtype="S32"))
            payload = _merge_particle_payloads(dynamic_payload, static_payload)
            payload["particle_ids"] = _to_s32(
                np.concatenate((dynamic_ids, static_ids)) if len(static_ids) else dynamic_ids)
            payload["source_dataset"] = np.asarray(ds, dtype=object)
            payload["frame_id"] = np.asarray(fr, dtype=object)
            xmf_path = particle_xmf.get(fr)
            if xmf_path is None:
                raise FileNotFoundError(
                    f"Missing particle XMF for frame {fr} in case {ds}; expected "
                    f"static_airfoil_pfield.*.xmf alongside {pin.name}"
                )
            payload["xmf_time"] = np.asarray(read_xmf_time(xmf_path), dtype=np.float64)
            payload["source_xmf_path"] = np.asarray(str(xmf_path), dtype=object)
            payload["source_vtk_path"] = np.asarray(str(vtk.get(fr, "")), dtype=object)
            payload["source_input_h5_path"] = np.asarray(str(pin), dtype=object)
            payload["source_static_h5_path"] = np.asarray("" if pstatic is None else str(pstatic), dtype=object)
            payload["source_output_h5_path"] = np.asarray("", dtype=object)

            output_path = out_dir / f"{ds}__frame_{fr}.npz"
            np.savez_compressed(output_path, **payload)
            merged.append(output_path)

    merged = sorted(merged, key=lambda p: (p.parent.name, p.stem))
    print("[merge] total merged:", len(merged))
    if len(merged) == 0:
        msg = (
            "No paired frames were merged. "
            f"RAW_ROOT={RAW_ROOT}, DATASET_IDS={DATASET_IDS}, "
            f"DYNAMIC_PARTICLE_H5_PATTERN={DYNAMIC_PARTICLE_H5_PATTERN}, "
            f"STATIC_PARTICLE_H5_PATTERN={STATIC_PARTICLE_H5_PATTERN}."
        )
        if missing_dataset_dirs:
            msg += " Missing dataset directories include: " + ", ".join(missing_dataset_dirs[:8])
        raise RuntimeError(msg)
    return merged


# The merged frame contains many source quantities, but only seven are propagated as the particle state. This function extracts those seven quantities with one scalar array per state component.
def _state_from_frame(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "particle_ids" in data:
        data["particle_ids"] = _to_s32(data["particle_ids"])
    xyz = as_xyz(data["particle_xyz"])
    gamma = as_xyz(data["Gamma_vec"])
    n = xyz.shape[0]
    sigma = _require_scalar(data["sigma"], n, "sigma")

    return {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "Gamma_x": gamma[:, 0],
        "Gamma_y": gamma[:, 1],
        "Gamma_z": gamma[:, 2],
        "sigma": sigma,
    }


# This routine creates the numerical input matrix for the particle model. One row represents one particle at the current time, and the columns are ordered according to PARTICLE_INPUT_FEATURES.
def _feature_matrix_from_state(
    state: Dict[str, np.ndarray],
    velocity: np.ndarray,
    gradient,
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
        "u_x": velocity[:n, 0],
        "u_y": velocity[:n, 1],
        "u_z": velocity[:n, 2],
        "gradU_xx": gradient[:n, 0],  "gradU_xy": gradient[:n, 1],  "gradU_xz": gradient[:n, 2],
        "gradU_yx": gradient[:n, 3],  "gradU_yy": gradient[:n, 4],  "gradU_yz": gradient[:n, 5],
        "gradU_zx": gradient[:n, 6],  "gradU_zy": gradient[:n, 7],  "gradU_zz": gradient[:n, 8],
    }

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


# The state matrix uses one particle per row and one state quantity per column. The normalization function computes channel statistics from training rows and applies those same statistics to every split.
def _state_matrix(state: Dict[str, np.ndarray], n: int) -> np.ndarray:
    return np.stack([state[k][:n] for k in STATE_NAMES], axis=1).astype(np.float32)


def _aligned_rollout_truth(frames: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return rollout states aligned to identities present in every frame."""
    if not frames:
        return np.zeros((0, 0, len(STATE_NAMES)), dtype=np.float32), np.zeros(0, dtype=object)
    first_ids = _to_s32(frames[0]["particle_ids"])
    shared_ids = np.char.decode(first_ids, "utf-8").tolist()
    for frame in frames[1:]:
        available = set(np.char.decode(_to_s32(frame["particle_ids"]), "utf-8").tolist())
        shared_ids = [identity for identity in shared_ids if identity in available]
    aligned = []
    for frame in frames:
        frame_ids = np.char.decode(_to_s32(frame["particle_ids"]), "utf-8")
        lookup = {identity: index for index, identity in enumerate(frame_ids)}
        indices = np.asarray([lookup[identity] for identity in shared_ids], dtype=np.int64)
        state = {name: np.asarray(values)[indices] for name, values in frame["state"].items()}
        aligned.append(_state_matrix(state, len(shared_ids)))
    return np.stack(aligned, axis=0).astype(np.float32), np.asarray(shared_ids, dtype=str)

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


def _load_and_track_case(case: str, plist: List[Path], meta: Dict[str, Any]):
    """Load one case's merged frames and attach persistent S32 identities."""
    frames = []
    for path in plist:
        with np.load(path, allow_pickle=True) as archive:
            data = {key: archive[key] for key in archive.files}
        state = _state_from_frame(data)
        velocity = as_xyz(data["velocity"])
        ids = _to_s32(data.get("particle_ids", [f"row:{j}" for j in range(len(state["x"]))]))
        if len(ids) != len(state["x"]):
            raise ValueError(f"Particle ID count does not match state rows in {path}")
        if "xmf_time" not in data:
            raise ValueError(f"Merged frame {path} has no XMF time metadata; rerun scripts/preprocess.py")
        xmf_time = float(np.asarray(data["xmf_time"]).reshape(-1)[0])
        frames.append({
            "frame_id": str(np.asarray(data["frame_id"]).reshape(-1)[0]),
            "xmf_time": xmf_time,
            "physical_time": xmf_time * float(meta["dt"]),
            "state": state,
            "particle_ids": ids,
            "static_mask": _require_scalar(data["static"], len(state["x"]), "static") > 0.5,
            "velocity": velocity,
            "velocity_gradient_x": as_xyz(data["velocity_gradient_x"]),
            "velocity_gradient_y": as_xyz(data["velocity_gradient_y"]),
            "velocity_gradient_z": as_xyz(data["velocity_gradient_z"]),
            "path": str(path),
            "vtk_path": str(np.asarray(data.get("source_vtk_path", "")).reshape(-1)[0]),
        })
        del data
    if len(frames) < 2:
        return frames
    times = np.asarray([float(frame["xmf_time"]) for frame in frames], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"XMF times for case {case} must be finite and strictly increasing; got {times.tolist()}")
    span = float(times[-1] - times[0])
    for frame in frames:
        frame["phase"] = 0.0 if span <= 0 else (float(frame["xmf_time"]) - float(times[0])) / span
    _assign_advected_position_tracks(frames, case)
    return frames


def _scan_pairs(by_case, meta_by_case, field_files_by_case=None):
    """Count tracked transitions one case at a time, releasing frames after each case."""
    pair_ranges = []
    field_available = []
    chain_lengths = {}
    start = 0
    print(f"[mem] scan start: RSS={_rss_gb():.2f} GB", flush=True)
    for case in sorted(by_case):
        frames = _load_and_track_case(case, by_case[case], meta_by_case[case])
        try:
            longest_chain = 0
            current_chain = 0
            for current, following in zip(frames, frames[1:]):
                _, _, common_ids = match_particle_identities(
                    current["particle_ids"], following["particle_ids"])
                count = len(common_ids)
                if count:
                    end = start + count
                    pair_ranges.append((case, current["frame_id"], following["frame_id"], start, end, count))
                    entries = [] if field_files_by_case is None else field_files_by_case.get(case, [])
                    matched_field = match_task2_xmf_time(entries, float(current["xmf_time"]))
                    field_available.append(matched_field is not None)
                    start = end
                    current_chain += 1
                else:
                    longest_chain = max(longest_chain, current_chain)
                    current_chain = 0
            longest_chain = max(longest_chain, current_chain)
            chain_lengths[case] = longest_chain
        finally:
            del frames
            gc.collect()
        print(f"[mem] after case={case}: RSS={_rss_gb():.2f} GB", flush=True)
        _check_rss_limit(case)
    return pair_ranges, np.asarray(field_available, dtype=bool), chain_lengths


def _fill_one_case(case, case_plist, case_meta, case_pairs, field_entries,
                   out_arrays, field_query_coords, targets_velocity_field,
                   field_query_mask, ids_concat, pair_contexts, field_bounds):
    """Build one case and write each transition directly into its global slices."""
    frames = _load_and_track_case(case, case_plist, case_meta)
    try:
        frame_index = {str(frame["frame_id"]): index for index, frame in enumerate(frames)}
        if len(frames) >= 2:
            aligned_states, rollout_ids = _aligned_rollout_truth(frames)
            rollout_ids = _to_s32(rollout_ids)
            rollout_phases = np.asarray([frame["phase"] for frame in frames], dtype=np.float32)
            rollout = (aligned_states, rollout_ids, rollout_phases,
                       np.asarray(float(case_meta["dt"]), dtype=np.float32))
        else:
            rollout = None

        for pair_id, global_start, global_end, expected_n, has_field, current_frame_id in case_pairs:
            current_index = frame_index[str(current_frame_id)]
            current, following = frames[current_index], frames[current_index + 1]
            idx0, idx1, common_ids = match_particle_identities(
                current["particle_ids"], following["particle_ids"])
            n = len(common_ids)
            if n != expected_n:
                raise RuntimeError(f"Pair-count scan/fill mismatch in {case} pair {pair_id}: {expected_n} != {n}")
            common_ids = _to_s32(common_ids)
            ids0, ids1 = current["particle_ids"], following["particle_ids"]
            s0 = {key: np.asarray(value)[idx0] if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(ids0) else value
                  for key, value in current["state"].items()}
            s1 = {key: np.asarray(value)[idx1] if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(ids1) else value
                  for key, value in following["state"].items()}
            curr = {key: (np.asarray(value)[idx0] if isinstance(value, np.ndarray) and value.ndim > 0
                          and len(value) == len(ids0) else value) for key, value in current.items()}
            nxt = {key: (np.asarray(value)[idx1] if isinstance(value, np.ndarray) and value.ndim > 0
                         and len(value) == len(ids1) else value) for key, value in following.items()}
            curr_xyz = np.stack([s0["x"], s0["y"], s0["z"]], axis=1)
            geometry = _particle_geometry_features(curr_xyz, str(curr.get("vtk_path", "")), n)
            gradient = np.concatenate([
                np.asarray(curr.get("velocity_gradient_x", np.zeros((n, 3))), dtype=np.float32),
                np.asarray(curr.get("velocity_gradient_y", np.zeros((n, 3))), dtype=np.float32),
                np.asarray(curr.get("velocity_gradient_z", np.zeros((n, 3))), dtype=np.float32),
            ], axis=1)
            velocity_current = np.asarray(curr["velocity"], dtype=np.float32)
            x_feat = _feature_matrix_from_state(
                s0, velocity_current, gradient, n, float(curr["phase"]),
                float(case_meta["aoa_deg"]), np.asarray(case_meta["freestream"], dtype=np.float64), geometry)
            state0, state1 = _state_matrix(s0, n), _state_matrix(s1, n)
            delta = state1 - state0
            delta_u = np.asarray(nxt["velocity"], dtype=np.float32) - velocity_current
            dt = float(following["physical_time"]) - float(current["physical_time"])
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError(f"Non-increasing XMF time for case={case}, frames {current['frame_id']}->{following['frame_id']}")
            physics_dx = (dt * velocity_current).astype(np.float32)
            physics_dgamma = (dt * np.einsum(
                "nij,nj->ni", gradient.reshape(n, 3, 3), state0[:, 3:6])).astype(np.float32)
            residual = np.concatenate([
                delta[:, :3] - physics_dx, delta[:, 3:6] - physics_dgamma,
                delta[:, 6:7], delta_u,
            ], axis=1).astype(np.float32)
            target = np.concatenate([delta, delta_u], axis=1).astype(np.float32)
            out_arrays["X"][global_start:global_end] = x_feat
            out_arrays["Y_delta"][global_start:global_end] = target
            out_arrays["Y_residual"][global_start:global_end] = residual
            out_arrays["Y_next"][global_start:global_end] = state1
            ids_concat[global_start:global_end] = common_ids

            decoded_common = np.char.decode(common_ids, "utf-8")
            if np.all(np.char.find(decoded_common, ":id:") >= 0):
                correspondence = "matched_source_particle_ids"
            elif np.all((np.char.find(decoded_common, "d:") >= 0)
                        | (np.char.find(decoded_common, "s:") >= 0)
                        | (np.char.find(decoded_common, "sx:") >= 0)
                        | (np.char.find(decoded_common, ":id:") >= 0)):
                correspondence = "inferred_advected_position_tracking"
            else:
                correspondence = "tagged_source_row_index_fallback"

            field_match = match_task2_xmf_time(field_entries, float(current["xmf_time"])) if has_field else None
            field_time, field_path = field_match if field_match is not None else (None, None)
            if field_path is not None:
                coords, values = read_field_grid_h5(field_path)
                n_query = min(len(coords), int(MAX_FIELD_QUERY_POINTS))
                selected = (np.linspace(0, len(coords) - 1, n_query, dtype=np.int64)
                            if len(coords) > n_query else np.arange(n_query, dtype=np.int64))
                if n_query:
                    field_query_coords[pair_id, :n_query] = coords[selected].astype(np.float32)
                    targets_velocity_field[pair_id, :n_query] = values[selected].astype(np.float32)
                    field_query_mask[pair_id, :n_query] = True
                    field_bounds[pair_id, 0] = np.min(coords, axis=0)
                    field_bounds[pair_id, 1] = np.max(coords, axis=0)
            pair_contexts[pair_id] = {
                "case": case, "frame_t": current["frame_id"], "frame_tp1": following["frame_id"],
                "xmf_time_t": float(current["xmf_time"]), "xmf_time_tp1": float(following["xmf_time"]),
                "physical_time_t": float(current["physical_time"]), "physical_time_tp1": float(following["physical_time"]),
                "task2_xmf_time": None if field_time is None else float(field_time),
                "task2_physical_time": None if field_time is None else float(field_time) * float(case_meta["dt"]),
                "task2_frame_id": "" if field_path is None else frame_id(field_path),
                "task2_field_path": "" if field_path is None else str(field_path),
                "start": int(global_start), "end": int(global_end), "n_particles": n,
                "phase_t": float(current["phase"]), "phase_tp1": float(following["phase"]),
                "aoa_deg": float(case_meta["aoa_deg"]),
                "freestream": [float(v) for v in np.asarray(case_meta["freestream"]).reshape(-1)],
                "dt": dt, "vtk_path": str(current.get("vtk_path", "")),
                "vtk_path_tp1": str(following.get("vtk_path", "")),
                "phase_delta": float(following["phase"] - current["phase"]),
                "correspondence_source": correspondence,
                "advected_track_match_fraction": float(following.get("advected_match_fraction", 1.0)),
            }
        return rollout
    finally:
        del frames


def _fill_pairs(by_case, pair_ranges, field_available, out_arrays, field_query_coords,
                targets_velocity_field, field_query_mask, ids_concat, meta_by_case,
                field_files_by_case, pair_contexts, field_bounds):
    """Fill preallocated arrays serially, releasing each case before continuing."""
    pairs_by_case = defaultdict(list)
    for pair_id, row in enumerate(pair_ranges):
        pairs_by_case[str(row[0])].append((pair_id, int(row[3]), int(row[4]), int(row[5]),
                                           bool(field_available[pair_id]), str(row[1])))
    rollout_by_case = {}
    for case in sorted(by_case):
        rollout = _fill_one_case(
            case, by_case[case], meta_by_case[case], pairs_by_case[case],
            field_files_by_case.get(case, []), out_arrays, field_query_coords,
            targets_velocity_field, field_query_mask, ids_concat, pair_contexts, field_bounds)
        if rollout is not None:
            rollout_by_case[case] = rollout
        gc.collect()
        print(f"[mem] after case={case}: RSS={_rss_gb():.2f} GB", flush=True)
        _check_rss_limit(case)
    return rollout_by_case


# This is the main dataset assembly routine. It groups merged frames by case, links each current particle frame to the matching Task 2 field, forms consecutive particle transitions, constructs both raw and physics-residual targets, and finally saves all representations together.
def _build_particle_evolution_dataset_legacy(merged: List[Path]) -> Path:
    global ACTIVE_FIELD_QUERY_BOUNDS
    by_case: Dict[str, List[Path]] = {}
    for p in merged:
        by_case.setdefault(p.parent.name, []).append(p)
    for c in by_case:
        by_case[c] = sorted(by_case[c], key=lambda x: frame_id(x))

    # Task-2 output is sparse and may use different filename indices, so index each field by its XMF simulation time.
    field_root = _resolve_field_root()
    field_files_by_case: Dict[str, List[Tuple[float, Path]]] = {}
    for case in sorted(by_case.keys()):
        case_field_root = field_root / case
        if not case_field_root.exists():
            field_files_by_case[case] = []
            continue
        entries = []
        seen_times: Dict[float, Path] = {}
        for path in sorted(case_field_root.glob(FIELD_H5_PATTERN)):
            xmf_path = path.with_suffix(".xmf")
            if not xmf_path.is_file():
                raise FileNotFoundError(f"Task-2 field {path} has no companion XMF: {xmf_path}")
            validate_task2_xmf_layout(xmf_path, path)
            field_time = read_xmf_time(xmf_path)
            duplicate = next((time for time in seen_times
                              if np.isclose(time, field_time, rtol=0.0, atol=XMF_TIME_MATCH_TOLERANCE)), None)
            if duplicate is not None:
                raise ValueError(
                    f"Task-2 case {case} has multiple fields at XMF time {field_time}: "
                    f"{seen_times[duplicate]} and {path}"
                )
            seen_times[field_time] = path
            entries.append((field_time, path))
        field_files_by_case[case] = sorted(entries, key=lambda item: item[0])
        print(f"[field] {case}: fdom frames={len(entries)}; XMF times="
              f"{[time for time, _ in field_files_by_case[case]][:8]}")

    if not os.environ.get("FIELD_QUERY_BOUNDS", "").strip():
        ACTIVE_FIELD_QUERY_BOUNDS = _infer_training_field_bounds(field_files_by_case)
    inferred_bounds = _field_query_bounds()
    print(f"[field] query domain from training meshes: {inferred_bounds}")

    def field_grid_for_time(case: str, task1_time: float):
        match = match_task2_xmf_time(field_files_by_case.get(case, []), task1_time)
        if match is None:
            return None
        field_time, path = match
        if not np.isclose(field_time, task1_time, rtol=0.0, atol=XMF_TIME_MATCH_TOLERANCE):
            raise AssertionError("Task-1/Task-2 XMF time match exceeded configured tolerance")
        return read_field_grid_h5(path), field_time, path

    all_cases = sorted(by_case.keys())
    _validate_case_split(all_cases)

    # Load the merged particle frames for one case before forming transitions. Particle identities are established across the complete sequence before the consecutive pairs are created.
    case_frames: Dict[str, List[Dict[str, object]]] = {}
    for case in all_cases:
        meta = _case_meta(case)
        plist = by_case[case]
        T = len(plist)
        if T < 2:
            print(f"[warn] case={case} has <2 frames; skipping")
            continue

        fr_list: List[Dict[str, object]] = []
        for p in plist:
            with np.load(p, allow_pickle=True) as d:
                data = {k: d[k] for k in d.files}
            state = _state_from_frame(data)
            velocity = as_xyz(data["velocity"])
            grad_x = as_xyz(data["velocity_gradient_x"])
            grad_y = as_xyz(data["velocity_gradient_y"])
            grad_z = as_xyz(data["velocity_gradient_z"])
            fr = str(np.asarray(data["frame_id"]).reshape(-1)[0])
            if "xmf_time" not in data:
                raise ValueError(f"Merged frame {p} has no XMF time metadata; rerun scripts/preprocess.py")
            xmf_time = float(np.asarray(data["xmf_time"]).reshape(-1)[0])
            vtk_path = str(np.asarray(data.get("source_vtk_path", "")).reshape(-1)[0])
            particle_ids = _to_s32(data.get("particle_ids", [f"row:{j}" for j in range(len(state["x"]))]))
            if len(particle_ids) != len(state["x"]):
                raise ValueError(f"Particle ID count does not match state rows in {p}")
            static_mask = _require_scalar(data["static"], len(state["x"]), "static") > 0.5
            fr_list.append(
                {
                    "frame_id": fr,
                    "xmf_time": xmf_time,
                    # XMF Time is the simulation-step coordinate; case metadata converts steps to physical seconds.
                    "physical_time": xmf_time * float(meta["dt"]),
                    "state": state,
                    "particle_ids": particle_ids,
                    "static_mask": static_mask,
                    "velocity": velocity,
                    "velocity_gradient_x": grad_x,
                    "velocity_gradient_y": grad_y,
                    "velocity_gradient_z": grad_z,
                    "path": str(p),
                    "vtk_path": vtk_path,
                }
            )
        xmf_times = np.asarray([float(frame["xmf_time"]) for frame in fr_list], dtype=np.float64)
        if not np.all(np.isfinite(xmf_times)) or np.any(np.diff(xmf_times) <= 0.0):
            raise ValueError(
                f"XMF times for case {case} must be finite and strictly increasing; "
                f"got {xmf_times.tolist()}"
            )
        time_span = float(xmf_times[-1] - xmf_times[0])
        for frame in fr_list:
            frame["phase"] = 0.0 if time_span <= 0.0 else (
                float(frame["xmf_time"]) - float(xmf_times[0])
            ) / time_span
        _assign_advected_position_tracks(fr_list, case)
        case_frames[case] = fr_list

    # These lists temporarily hold one array per pair. Keeping the pairs separate during construction lets the code retain the exact row boundaries needed for split and frame-level bookkeeping.
    rows_x: List[np.ndarray] = []
    rows_delta: List[np.ndarray] = []
    rows_residual: List[np.ndarray] = []
    rows_next: List[np.ndarray] = []
    rows_particle_ids: List[np.ndarray] = []
    field_query_coords_by_pair: List[np.ndarray] = []
    field_velocity_by_pair: List[np.ndarray] = []
    skipped_pairs_missing_field = 0

    pair_ranges: List[Tuple] = []
    pair_contexts: List[Dict[str, object]] = []

    rollout_cases: List[str] = []
    rollout_true_states: List[np.ndarray] = []
    rollout_true_particle_ids: List[np.ndarray] = []
    rollout_phases: List[np.ndarray] = []
    rollout_dts: List[float] = []

    start = 0

    # Every consecutive pair represents one supervised time step. The input comes from the current frame and the target is obtained by comparing the matched particles with their values in the next frame.
    for case in sorted(case_frames.keys()):
        meta = _case_meta(case)
        fr_list = case_frames[case]
        T = len(fr_list)

        # Keep a copy of the ground-truth state sequence for this case so that later rollout analysis has access to the original trajectory rather than only the flattened one-step pairs.
        aligned_states, shared_ids = _aligned_rollout_truth(fr_list)
        phase_seq = [float(fr["phase"]) for fr in fr_list]

        rollout_cases.append(case)
        rollout_true_states.append(aligned_states)
        rollout_true_particle_ids.append(_to_s32(shared_ids))
        rollout_phases.append(np.asarray(phase_seq, dtype=np.float32))
        rollout_dts.append(float(meta["dt"]))

        # Construct the one-step learning examples from consecutive frames in chronological order.
        for i in range(T - 1):
            curr = fr_list[i]
            nxt = fr_list[i + 1]
            field_match = field_grid_for_time(case, float(curr["xmf_time"]))
            if field_match is None:
                skipped_pairs_missing_field += 1
                # The particle transition is still retained when the matching Task 2 field file is absent. The field arrays for this pair are left empty, and the field mask later marks the pair as containing no valid field samples.
                field_grid = None
                field_time = None
                field_path = ""
            else:
                field_grid, field_time, matched_field_path = field_match
                field_path = str(matched_field_path)

            s0 = curr["state"]
            s1 = nxt["state"]

            ids0, ids1 = curr["particle_ids"], nxt["particle_ids"]
            idx0, idx1, common_ids = match_particle_identities(ids0, ids1)
            n = len(common_ids)
            if n <= 0:
                continue

            # Apply the current-frame correspondence indices to every particle-sized state array before forming the learning features.
            s0 = {key: np.asarray(value)[idx0] if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(ids0) else value
                  for key, value in s0.items()}
            s1 = {key: np.asarray(value)[idx1] if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(ids1) else value
                  for key, value in s1.items()}
            curr = {key: (np.asarray(value)[idx0] if isinstance(value, np.ndarray) and value.ndim > 0
                          and len(value) == len(ids0) else value) for key, value in curr.items()}
            nxt = {key: (np.asarray(value)[idx1] if isinstance(value, np.ndarray) and value.ndim > 0
                         and len(value) == len(ids1) else value) for key, value in nxt.items()}
            curr_xyz = np.stack([s0["x"], s0["y"], s0["z"]], axis=1)
            geom_feat = _particle_geometry_features(curr_xyz, str(curr.get("vtk_path", "")), n)

            grad_x = np.asarray(curr.get("velocity_gradient_x", np.zeros((n,3))), dtype=np.float32)
            grad_y = np.asarray(curr.get("velocity_gradient_y", np.zeros((n,3))), dtype=np.float32)
            grad_z = np.asarray(curr.get("velocity_gradient_z", np.zeros((n,3))), dtype=np.float32)
            gradient = np.concatenate([grad_x, grad_y, grad_z], axis=1)

            x_feat = _feature_matrix_from_state(
                state=s0,
                velocity=np.asarray(curr["velocity"], dtype=np.float32),
                gradient=gradient,
                n=n,
                phase=float(curr["phase"]),
                aoa_deg=float(meta["aoa_deg"]),
                freestream=np.asarray(meta["freestream"], dtype=np.float64),
                geom_feat=geom_feat,
            )

            st0 = _state_matrix(s0, n)
            st1 = _state_matrix(s1, n)
            delta = st1 - st0
            # The velocity change is stored as an additional target because velocity itself is part of the input rather than the propagated seven-component state.
            velocity_current = np.asarray(curr["velocity"], dtype=np.float32)
            velocity_next = np.asarray(nxt["velocity"], dtype=np.float32)
            delta_u = velocity_next - velocity_current
            dt = float(nxt["physical_time"]) - float(curr["physical_time"])
            if not np.isfinite(dt) or dt <= 0.0:
                raise ValueError(
                    f"Non-increasing XMF time for case={case}, frames "
                    f"{curr['frame_id']}->{nxt['frame_id']}"
                )
            gamma0 = st0[:, 3:6]
            grad_tensor = gradient.reshape(n, 3, 3).astype(np.float32)
            # Position change expected from first-order particle advection is dt times the current particle velocity.
            physics_dx = (dt * velocity_current).astype(np.float32)
            # The vortex-strength contribution uses the current velocity-gradient tensor acting on the current particle strength.
            physics_dgamma = (dt * np.einsum("nij,nj->ni", grad_tensor, gamma0)).astype(np.float32)
            residual = np.concatenate(
                [
                    delta[:, :3] - physics_dx,
                    delta[:, 3:6] - physics_dgamma,
                    delta[:, 6:7],
                    delta_u,
                ],
                axis=1,
            ).astype(np.float32)
            target = np.concatenate([delta, delta_u], axis=1).astype(np.float32)

            end = start + n
            pair_ranges.append((case, curr["frame_id"], nxt["frame_id"], start, end, n))
            if all(":id:" in str(identity) for identity in common_ids):
                correspondence_source = "matched_source_particle_ids"
            elif all(("d:" in str(identity) or "s:" in str(identity) or "sx:" in str(identity)
                      or ":id:" in str(identity)) for identity in common_ids):
                correspondence_source = "inferred_advected_position_tracking"
            else:
                correspondence_source = "tagged_source_row_index_fallback"
            pair_contexts.append(
                {
                    "case": case,
                    "frame_t": curr["frame_id"],
                    "frame_tp1": nxt["frame_id"],
                    "xmf_time_t": float(curr["xmf_time"]),
                    "xmf_time_tp1": float(nxt["xmf_time"]),
                    "physical_time_t": float(curr["physical_time"]),
                    "physical_time_tp1": float(nxt["physical_time"]),
                    "task2_xmf_time": None if field_time is None else float(field_time),
                    "task2_physical_time": None if field_time is None else float(field_time) * float(meta["dt"]),
                    "task2_frame_id": "" if field_match is None else frame_id(matched_field_path),
                    "task2_field_path": field_path,
                    "start": start,
                    "end": end,
                    "n_particles": n,
                    "phase_t": float(curr["phase"]),
                    "phase_tp1": float(nxt["phase"]),
                    "aoa_deg": float(meta["aoa_deg"]),
                    "freestream": [float(v) for v in np.asarray(meta["freestream"]).reshape(-1)],
                    "dt": dt,
                    "vtk_path": str(curr.get("vtk_path", "")),
                    "vtk_path_tp1": str(nxt.get("vtk_path", "")),
                    "phase_delta": float(nxt["phase"] - curr["phase"]),
                    "correspondence_source": correspondence_source,
                    "advected_track_match_fraction": float(nxt.get("advected_match_fraction", 1.0)),
                }
            )

            rows_x.append(x_feat.astype(np.float32))
            rows_delta.append(target)
            rows_residual.append(residual)
            rows_next.append(st1.astype(np.float32))
            rows_particle_ids.append(_to_s32(common_ids))
            if field_grid is None:
                grid_coords = np.zeros((0, 3), dtype=np.float32)
                grid_velocity = np.zeros((0, 12), dtype=np.float32)
            else:
                grid_coords, grid_velocity = field_grid
            field_query_coords_by_pair.append(grid_coords.astype(np.float32))
            field_velocity_by_pair.append(grid_velocity.astype(np.float32))
            start = end

    if not rows_x:
        raise RuntimeError("No Task-1 pairs were built. Check frame availability and metadata.")

    # After all cases are processed, concatenate the per-pair arrays along the particle-row dimension. pair_ranges retains the boundaries needed to recover the original pair structure.
    X = np.concatenate(rows_x, axis=0).astype(np.float32)
    Y_delta = np.concatenate(rows_delta, axis=0).astype(np.float32)
    Y_residual = np.concatenate(rows_residual, axis=0).astype(np.float32)
    Y_next = np.concatenate(rows_next, axis=0).astype(np.float32)

    n_pairs = len(pair_ranges)

    # Field samples remaining after filtering need not have the same count for every pair, so they are stored in a fixed-size array with a Boolean mask indicating which entries are real.
    field_query_coords = np.zeros((n_pairs, MAX_FIELD_QUERY_POINTS, 3), dtype=np.float32)
    targets_velocity_field = np.zeros((n_pairs, MAX_FIELD_QUERY_POINTS, 12), dtype=np.float32)

    field_query_mask = np.zeros((n_pairs, MAX_FIELD_QUERY_POINTS), dtype=bool)
    for pair_id, (query_coords, velocity_values) in enumerate(zip(field_query_coords_by_pair, field_velocity_by_pair)):
        n_query = min(int(query_coords.shape[0]), int(MAX_FIELD_QUERY_POINTS))
        if n_query <= 0:
            continue
        if query_coords.shape[0] > n_query:
            picked = np.linspace(0, query_coords.shape[0] - 1, n_query, dtype=np.int64)
        else:
            picked = np.arange(n_query, dtype=np.int64)
        field_query_coords[pair_id, :n_query, :] = query_coords[picked, :]
        targets_velocity_field[pair_id, :n_query, :] = velocity_values[picked, :]
        field_query_mask[pair_id, :n_query] = True

    # The split is first defined at the pair level and is then expanded into row indices. This preserves complete particle transitions inside each split.
    pair_split_train_case = np.array([i for i, r in enumerate(pair_ranges) if _case_split_label(r[0]) == "train"], dtype=np.int64)
    pair_split_val = np.array([i for i, r in enumerate(pair_ranges) if _case_split_label(r[0]) == "val"], dtype=np.int64)
    pair_split_test = np.array([i for i, r in enumerate(pair_ranges) if _case_split_label(r[0]) == "test"], dtype=np.int64)
    pair_split_train, pair_split_train_id_val = _train_id_val_id_split_by_case(
        pair_ranges, pair_split_train_case
    )

    # Reserve selected Task-2 times only from field supervision; their Task-1 transitions remain available for evolution training.
    field_train_candidates = [int(pid) for pid in pair_split_train
                              if np.asarray(field_query_mask[int(pid)], dtype=bool).any()]
    field_candidates_by_case: Dict[str, List[int]] = defaultdict(list)
    for pid in field_train_candidates:
        field_candidates_by_case[str(pair_ranges[pid][0])].append(pid)
    field_superres_pair_ids = []
    stride = max(int(FIELD_SUPERRESOLUTION_STRIDE), 2)
    offset = min(max(int(FIELD_SUPERRESOLUTION_OFFSET), 0), stride - 1)
    for case, ids in sorted(field_candidates_by_case.items()):
        ids.sort(key=lambda pid: float(pair_contexts[pid]["xmf_time_t"]))
        if len(ids) > 1:
            field_superres_pair_ids.extend(ids[offset::stride])
    field_superres_pair_ids = np.asarray(sorted(field_superres_pair_ids), dtype=np.int64)
    field_stats_pair_ids = np.setdiff1d(pair_split_train, field_superres_pair_ids, assume_unique=False)

    train_rows = _rows_from_pair_ids(pair_ranges, pair_split_train)
    val_rows = _rows_from_pair_ids(pair_ranges, pair_split_val)
    test_rows = _rows_from_pair_ids(pair_ranges, pair_split_test)

    # The normalization statistics are fitted only to training rows. Validation and test data are transformed with those fixed training statistics rather than influencing them.
    in_mean, in_std, Xn = _normalize_channels_rows(X, train_rows)
    raw_delta_mean, raw_delta_std, Yn_delta_raw = _normalize_channels_rows(Y_delta, train_rows)
    residual_mean, residual_std, Yn_residual = _normalize_channels_rows(Y_residual, train_rows)

    train_field_values = targets_velocity_field[field_stats_pair_ids][field_query_mask[field_stats_pair_ids]]
    if train_field_values.size == 0:
        raise RuntimeError("No field-reconstruction query points were available in the training split.")
    train_field_values = train_field_values[np.isfinite(train_field_values).all(axis=1)]
    if train_field_values.size == 0:
        raise RuntimeError("No finite field-reconstruction targets were available in the training split.")
    # Use the configured floor so a nearly constant field channel does not produce an excessively large normalized value from division by an almost-zero standard deviation.
    field_mean = np.mean(train_field_values, axis=0, keepdims=True).astype(np.float32)
    field_std = np.maximum(np.std(train_field_values, axis=0, keepdims=True), FIELD_STD_FLOOR).astype(np.float32)
    targets_velocity_field_norm = ((targets_velocity_field - field_mean.reshape(1, 1, 12)) / field_std.reshape(1, 1, 12)).astype(np.float32)

    next_mean = np.mean(Y_next[train_rows], axis=0, keepdims=True)
    next_std = np.maximum(np.std(Y_next[train_rows], axis=0, keepdims=True), 1e-8)
    Yn_next = ((Y_next - next_mean) / next_std).astype(np.float32)

    # Spatial coordinate extents are stored separately from the channel-wise normalization because downstream models may need the physical coordinate range for their own coordinate mapping.
    coord_cols = [PARTICLE_INPUT_FEATURES.index(k) for k in ("x", "y", "z")]
    particle_coords = X[train_rows][:, coord_cols]
    train_field_coord_chunks = [field_query_coords_by_pair[int(pid)] for pid in field_stats_pair_ids
                                if len(field_query_coords_by_pair[int(pid)])]
    valid_field_coords = (np.concatenate(train_field_coord_chunks, axis=0)
                          if train_field_coord_chunks else np.zeros((0, 3), dtype=np.float32))
    if valid_field_coords.size:
        coord_source = np.concatenate([particle_coords, valid_field_coords], axis=0)
    else:
        coord_source = particle_coords
    coord_min = np.min(coord_source, axis=0).astype(np.float32)
    coord_span = np.maximum(np.ptp(coord_source, axis=0), 1e-8).astype(np.float32)
    coord_max = (coord_min + coord_span).astype(np.float32)
    particle_coord_min = np.min(particle_coords, axis=0).astype(np.float32)
    particle_coord_max = np.max(particle_coords, axis=0).astype(np.float32)
    if valid_field_coords.size:
        # If there are no valid field coordinates, retain NaN bounds to make that missing field information explicit in the saved metadata.
        field_coord_min = np.min(valid_field_coords, axis=0).astype(np.float32)
        field_coord_max = np.max(valid_field_coords, axis=0).astype(np.float32)
    else:
        field_coord_min = np.full(3, np.nan, dtype=np.float32)
        field_coord_max = np.full(3, np.nan, dtype=np.float32)

    # Store the raw and normalized arrays together with their names, split indices, metadata, and normalization statistics so downstream training and analysis can reproduce the exact representation created here.
    out_path = OUT_ROOT / "particle_evolution_dataset.npz"
    np.savez_compressed(
        out_path,
        inputs_t=X,
        particle_ids_t=np.concatenate(rows_particle_ids) if rows_particle_ids else np.zeros(0, dtype="S32"),
        particle_ids_tp1=np.concatenate(rows_particle_ids) if rows_particle_ids else np.zeros(0, dtype="S32"),
        targets_delta=Y_delta,
        targets_residual=Y_residual,
        query_coords=field_query_coords,
        targets_velocity_field=targets_velocity_field,
        field_query_mask=field_query_mask,
        # Keep the field query policy in the saved dataset so later code can identify which physical region and sampling rule produced these targets.
        field_query_source=np.asarray("task2_static_airfoil_fdom_grid_filtered", dtype=object),
        field_query_bounds=np.asarray(_field_query_bounds() if _field_query_bounds() is not None else (), dtype=np.float32),
        field_std_floor=np.asarray(FIELD_STD_FLOOR, dtype=np.float32),
        field_root=np.asarray(str(field_root), dtype=object),
        targets_next_state=Y_next,
        inputs_t_norm=Xn,
        targets_delta_norm=Yn_delta_raw,
        targets_residual_norm=Yn_residual,
        targets_velocity_field_norm=targets_velocity_field_norm,
        targets_next_state_norm=Yn_next,
        feature_names=np.asarray(PARTICLE_INPUT_FEATURES, dtype=object),
        state_names=np.asarray(STATE_NAMES, dtype=object),
        target_names=np.asarray(TARGET_DELTA_NAMES, dtype=object),
        field_target_names=np.asarray(FIELD_TARGET_NAMES, dtype=object),
        pair_ranges=np.asarray(pair_ranges, dtype=object),
        pair_contexts=np.asarray(pair_contexts, dtype=object),
        train_pair_ids=pair_split_train,
        val_pair_ids=pair_split_val,
        test_pair_ids=pair_split_test,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        rollout_cases=np.asarray(rollout_cases, dtype=object),
        rollout_true_states=np.asarray(rollout_true_states, dtype=object),
        rollout_true_particle_ids=np.asarray(rollout_true_particle_ids, dtype=object),
        rollout_phases=np.asarray(rollout_phases, dtype=object),
        rollout_dts=np.asarray(rollout_dts, dtype=np.float32),
        train_id_val_pair_ids=pair_split_train_id_val,
        field_superres_pair_ids=field_superres_pair_ids,
        field_superres_stride=np.asarray(stride, dtype=np.int64),
        field_superres_offset=np.asarray(offset, dtype=np.int64),
        train_cases=np.asarray(TRAIN_CASES, dtype=object),
        val_cases=np.asarray(VAL_CASES, dtype=object),
        test_cases=np.asarray(TEST_CASES, dtype=object),
        case_metadata=np.asarray(CASE_METADATA, dtype=object),
        use_geometry_channels=np.asarray(USE_GEOMETRY_CHANNELS),
        geometry_channel_names=np.asarray(GEOMETRY_CHANNEL_NAMES, dtype=object),
        in_mean=in_mean.astype(np.float32),
        in_std=in_std.astype(np.float32),
        out_mean=residual_mean.astype(np.float32),
        out_std=residual_std.astype(np.float32),
        residual_mean=residual_mean.astype(np.float32),
        residual_std=residual_std.astype(np.float32),
        raw_delta_mean=raw_delta_mean.astype(np.float32),
        raw_delta_std=raw_delta_std.astype(np.float32),
        field_mean=field_mean.astype(np.float32),
        field_std=field_std.astype(np.float32),
        next_mean=next_mean.astype(np.float32),
        next_std=next_std.astype(np.float32),
        coord_min=coord_min,
        coord_span=coord_span,
        coord_max=coord_max,
        particle_coord_min=particle_coord_min,
        particle_coord_max=particle_coord_max,
        field_coord_min=field_coord_min,
        field_coord_max=field_coord_max,
        max_field_query_points=np.asarray(MAX_FIELD_QUERY_POINTS, dtype=np.int64),
    )

    # Print the main dimensions and preprocessing settings at the end of the run so the generated dataset can be checked against the expected case and field counts.
    print("\n[task1] Particle evolution dataset built")
    print("  inputs_t shape            :", X.shape)
    print("  targets_delta shape       :", Y_delta.shape)
    print("  query_coords shape        :", field_query_coords.shape)
    print("  targets_velocity_field    :", targets_velocity_field.shape)
    print("  field query bounds        :", _field_query_bounds())
    print("  field queries per pair    :", int(field_query_mask.sum(axis=1).min()), float(field_query_mask.sum(axis=1).mean()), int(field_query_mask.sum(axis=1).max()))
    print("  field target mean/std     :", field_mean.reshape(-1).tolist(), field_std.reshape(-1).tolist())
    print("  pairs without field labels:", skipped_pairs_missing_field)
    print("  coord_min train only      :", coord_min.tolist())
    print("  coord_max train only      :", coord_max.tolist())
    print("  particle coord min/max    :", particle_coord_min.tolist(), particle_coord_max.tolist())
    print("  field coord min/max       :", field_coord_min.tolist(), field_coord_max.tolist())
    print("  targets_next_state shape  :", Y_next.shape)
    print("  n_pairs                   :", len(pair_ranges))
    print("  n_rollout_cases           :", len(rollout_cases))
    print("  feature_names             :", PARTICLE_INPUT_FEATURES)
    print("  use_geometry_channels     :", USE_GEOMETRY_CHANNELS)
    print("  target_names              :", TARGET_DELTA_NAMES)
    print("  split(train/internal-val/val/test) pairs:", len(pair_split_train), len(pair_split_train_id_val), len(pair_split_val), len(pair_split_test))
    print("  field temporal holdout pairs:", len(field_superres_pair_ids))
    print("  split(train/val/test) rows :", len(train_rows), len(val_rows), len(test_rows))
    print("  train/val/test cases      :", TRAIN_CASES, VAL_CASES, TEST_CASES)
    _print_geometry_channel_stats(X, PARTICLE_INPUT_FEATURES, "task1-delta")

    return out_path


def build_particle_evolution_dataset(merged: List[Path]) -> Path:
    """Build the final dataset with a count pass and bounded per-case fill pass."""
    global ACTIVE_FIELD_QUERY_BOUNDS
    if os.environ.get("PREPROC_STREAMING", "1") == "0":
        return _build_particle_evolution_dataset_legacy(merged)

    by_case: Dict[str, List[Path]] = defaultdict(list)
    for path in merged:
        by_case[path.parent.name].append(path)
    for case in by_case:
        by_case[case].sort(key=frame_id)
    all_cases = sorted(by_case)
    _validate_case_split(all_cases)
    meta_by_case = {case: _case_meta(case) for case in all_cases}

    field_root = _resolve_field_root()
    field_files_by_case: Dict[str, List[Tuple[float, Path]]] = {}
    for case in all_cases:
        entries = []
        for path in sorted((field_root / case).glob(FIELD_H5_PATTERN)):
            xmf_path = path.with_suffix(".xmf")
            if not xmf_path.is_file():
                raise FileNotFoundError(f"Task-2 field {path} has no companion XMF: {xmf_path}")
            validate_task2_xmf_layout(xmf_path, path)
            entries.append((read_xmf_time(xmf_path), path))
        entries.sort(key=lambda item: item[0])
        for previous, current in zip(entries, entries[1:]):
            if np.isclose(previous[0], current[0], rtol=0.0, atol=XMF_TIME_MATCH_TOLERANCE):
                raise ValueError(f"Task-2 case {case} has duplicate XMF time {current[0]}")
        field_files_by_case[case] = entries
        print(f"[field] {case}: fdom frames={len(entries)}; XMF times={[t for t, _ in entries][:8]}")
    if not os.environ.get("FIELD_QUERY_BOUNDS", "").strip():
        ACTIVE_FIELD_QUERY_BOUNDS = _infer_training_field_bounds(field_files_by_case)

    pair_ranges, field_available, chain_lengths = _scan_pairs(by_case, meta_by_case, field_files_by_case)
    if not pair_ranges:
        raise RuntimeError("No Task-1 pairs were built. Check frame availability and metadata.")
    n_pairs = len(pair_ranges)
    total_rows = sum(row[5] for row in pair_ranges)
    X = np.empty((total_rows, len(PARTICLE_INPUT_FEATURES)), dtype=np.float32)
    Y_delta = np.empty((total_rows, len(TARGET_DELTA_NAMES)), dtype=np.float32)
    Y_residual = np.empty_like(Y_delta)
    Y_next = np.empty((total_rows, 7), dtype=np.float32)
    ids_concat = np.empty(total_rows, dtype="S32")
    field_query_coords = np.zeros((n_pairs, MAX_FIELD_QUERY_POINTS, 3), dtype=np.float32)
    targets_velocity_field = np.zeros((n_pairs, MAX_FIELD_QUERY_POINTS, 12), dtype=np.float32)
    field_query_mask = np.zeros((n_pairs, MAX_FIELD_QUERY_POINTS), dtype=bool)
    field_bounds = np.full((n_pairs, 2, 3), np.nan, dtype=np.float32)
    pair_contexts: List[Dict[str, object]] = [None] * n_pairs
    out_arrays = {"X": X, "Y_delta": Y_delta, "Y_residual": Y_residual, "Y_next": Y_next}
    rollout_by_case = _fill_pairs(
        by_case, pair_ranges, field_available, out_arrays, field_query_coords,
        targets_velocity_field, field_query_mask, ids_concat, meta_by_case,
        field_files_by_case, pair_contexts, field_bounds)

    rollout_cases = sorted(rollout_by_case)
    rollout_true_states, rollout_true_particle_ids, rollout_phases, rollout_dts = [], [], [], []
    rollout_bytes = 0
    for case in rollout_cases:
        states, ids, phases, dt = rollout_by_case[case]
        rollout_true_states.append(states)
        rollout_true_particle_ids.append(_to_s32(ids))
        rollout_phases.append(phases)
        rollout_dts.append(float(np.asarray(dt)))
        rollout_bytes += states.nbytes + ids.nbytes + phases.nbytes
    if rollout_bytes > ROLLOUT_MEMORY_CAP_GB * 1e9:
        print(f"[mem] rollout arrays={rollout_bytes / 1e9:.3f} GB exceed cap="
              f"{ROLLOUT_MEMORY_CAP_GB:.3f} GB; retaining per-case arrays until archive write")
    else:
        print(f"[mem] rollout arrays={rollout_bytes / 1e9:.3f} GB; cap="
              f"{ROLLOUT_MEMORY_CAP_GB:.3f} GB")

    pair_split_train_case = np.asarray([i for i, row in enumerate(pair_ranges)
                                        if _case_split_label(row[0]) == "train"], dtype=np.int64)
    pair_split_val = np.asarray([i for i, row in enumerate(pair_ranges)
                                 if _case_split_label(row[0]) == "val"], dtype=np.int64)
    pair_split_test = np.asarray([i for i, row in enumerate(pair_ranges)
                                  if _case_split_label(row[0]) == "test"], dtype=np.int64)
    pair_split_train, pair_split_train_id_val = _train_id_val_id_split_by_case(pair_ranges, pair_split_train_case)
    field_train_candidates = [int(pid) for pid in pair_split_train if field_query_mask[int(pid)].any()]
    field_candidates_by_case: Dict[str, List[int]] = defaultdict(list)
    for pid in field_train_candidates:
        field_candidates_by_case[str(pair_ranges[pid][0])].append(pid)
    stride = max(int(FIELD_SUPERRESOLUTION_STRIDE), 2)
    offset = min(max(int(FIELD_SUPERRESOLUTION_OFFSET), 0), stride - 1)
    field_superres_pair_ids = []
    for case, pair_ids in sorted(field_candidates_by_case.items()):
        pair_ids.sort(key=lambda pid: float(pair_contexts[pid]["xmf_time_t"]))
        if len(pair_ids) > 1:
            field_superres_pair_ids.extend(pair_ids[offset::stride])
    field_superres_pair_ids = np.asarray(sorted(field_superres_pair_ids), dtype=np.int64)
    field_stats_pair_ids = np.setdiff1d(pair_split_train, field_superres_pair_ids)
    train_rows = _rows_from_pair_ids(pair_ranges, pair_split_train)
    val_rows = _rows_from_pair_ids(pair_ranges, pair_split_val)
    test_rows = _rows_from_pair_ids(pair_ranges, pair_split_test)
    in_mean, in_std, Xn = _normalize_channels_rows(X, train_rows)
    raw_delta_mean, raw_delta_std, Yn_delta_raw = _normalize_channels_rows(Y_delta, train_rows)
    residual_mean, residual_std, Yn_residual = _normalize_channels_rows(Y_residual, train_rows)
    train_field_values = targets_velocity_field[field_stats_pair_ids][field_query_mask[field_stats_pair_ids]]
    if not train_field_values.size:
        raise RuntimeError("No field-reconstruction query points were available in the training split.")
    train_field_values = train_field_values[np.isfinite(train_field_values).all(axis=1)]
    if not train_field_values.size:
        raise RuntimeError("No finite field-reconstruction targets were available in the training split.")
    field_mean = np.mean(train_field_values, axis=0, keepdims=True).astype(np.float32)
    field_std = np.maximum(np.std(train_field_values, axis=0, keepdims=True), FIELD_STD_FLOOR).astype(np.float32)
    targets_velocity_field_norm = ((targets_velocity_field - field_mean.reshape(1, 1, 12)) /
                                   field_std.reshape(1, 1, 12)).astype(np.float32)
    next_mean = np.mean(Y_next[train_rows], axis=0, keepdims=True)
    next_std = np.maximum(np.std(Y_next[train_rows], axis=0, keepdims=True), 1e-8)
    Yn_next = ((Y_next - next_mean) / next_std).astype(np.float32)

    coord_cols = [PARTICLE_INPUT_FEATURES.index(key) for key in ("x", "y", "z")]
    particle_coords = X[train_rows][:, coord_cols]
    valid_field_pair_ids = [int(pid) for pid in field_stats_pair_ids if np.isfinite(field_bounds[pid]).all()]
    if valid_field_pair_ids:
        field_coord_min = np.min(field_bounds[valid_field_pair_ids, 0], axis=0)
        field_coord_max = np.max(field_bounds[valid_field_pair_ids, 1], axis=0)
        coord_source = np.concatenate([particle_coords, field_coord_min[None], field_coord_max[None]], axis=0)
    else:
        field_coord_min = np.full(3, np.nan, dtype=np.float32)
        field_coord_max = np.full(3, np.nan, dtype=np.float32)
        coord_source = particle_coords
    coord_min = np.min(coord_source, axis=0).astype(np.float32)
    coord_span = np.maximum(np.ptp(coord_source, axis=0), 1e-8).astype(np.float32)
    coord_max = (coord_min + coord_span).astype(np.float32)
    particle_coord_min = np.min(particle_coords, axis=0).astype(np.float32)
    particle_coord_max = np.max(particle_coords, axis=0).astype(np.float32)
    field_coord_min = field_coord_min.astype(np.float32)
    field_coord_max = field_coord_max.astype(np.float32)

    out_path = OUT_ROOT / "particle_evolution_dataset.npz"
    np.savez_compressed(
        out_path, inputs_t=X, particle_ids_t=ids_concat, particle_ids_tp1=ids_concat,
        targets_delta=Y_delta, targets_residual=Y_residual, query_coords=field_query_coords,
        targets_velocity_field=targets_velocity_field, field_query_mask=field_query_mask,
        field_query_source=np.asarray("task2_static_airfoil_fdom_grid_filtered", dtype=object),
        field_query_bounds=np.asarray(_field_query_bounds() if _field_query_bounds() is not None else (), dtype=np.float32),
        field_std_floor=np.asarray(FIELD_STD_FLOOR, dtype=np.float32), field_root=np.asarray(str(field_root), dtype=object),
        targets_next_state=Y_next, inputs_t_norm=Xn, targets_delta_norm=Yn_delta_raw,
        targets_residual_norm=Yn_residual, targets_velocity_field_norm=targets_velocity_field_norm,
        targets_next_state_norm=Yn_next, feature_names=np.asarray(PARTICLE_INPUT_FEATURES, dtype=object),
        state_names=np.asarray(STATE_NAMES, dtype=object), target_names=np.asarray(TARGET_DELTA_NAMES, dtype=object),
        field_target_names=np.asarray(FIELD_TARGET_NAMES, dtype=object), pair_ranges=np.asarray(pair_ranges, dtype=object),
        pair_contexts=np.asarray(pair_contexts, dtype=object), train_pair_ids=pair_split_train,
        val_pair_ids=pair_split_val, test_pair_ids=pair_split_test, train_rows=train_rows,
        val_rows=val_rows, test_rows=test_rows, rollout_cases=np.asarray(rollout_cases, dtype=object),
        rollout_true_states=np.asarray(rollout_true_states, dtype=object),
        rollout_true_particle_ids=np.asarray(rollout_true_particle_ids, dtype=object),
        rollout_phases=np.asarray(rollout_phases, dtype=object), rollout_dts=np.asarray(rollout_dts, dtype=np.float32),
        train_id_val_pair_ids=pair_split_train_id_val, field_superres_pair_ids=field_superres_pair_ids,
        field_superres_stride=np.asarray(stride, dtype=np.int64), field_superres_offset=np.asarray(offset, dtype=np.int64),
        train_cases=np.asarray(TRAIN_CASES, dtype=object), val_cases=np.asarray(VAL_CASES, dtype=object),
        test_cases=np.asarray(TEST_CASES, dtype=object), case_metadata=np.asarray(CASE_METADATA, dtype=object),
        use_geometry_channels=np.asarray(USE_GEOMETRY_CHANNELS),
        geometry_channel_names=np.asarray(GEOMETRY_CHANNEL_NAMES, dtype=object), in_mean=in_mean,
        in_std=in_std, out_mean=residual_mean, out_std=residual_std, residual_mean=residual_mean,
        residual_std=residual_std, raw_delta_mean=raw_delta_mean, raw_delta_std=raw_delta_std,
        field_mean=field_mean, field_std=field_std, next_mean=next_mean, next_std=next_std,
        coord_min=coord_min, coord_span=coord_span, coord_max=coord_max,
        particle_coord_min=particle_coord_min, particle_coord_max=particle_coord_max,
        field_coord_min=field_coord_min, field_coord_max=field_coord_max,
        max_field_query_points=np.asarray(MAX_FIELD_QUERY_POINTS, dtype=np.int64))
    print(f"[task1] streaming dataset built: rows={total_rows}, pairs={n_pairs}, rollouts={len(rollout_cases)}")
    return out_path


# The script starts here when executed directly. It resolves the source roots, discovers and splits the cases, validates the metadata, creates the intermediate merged frames, and then builds the final particle-evolution dataset.
def main() -> None:
    global ACTIVE_CASE_METADATA
    global ACTIVE_CONDITIONING_CHANNEL_NAMES
    global RAW_ROOT
    global DATASET_IDS

    RAW_ROOT = _resolve_raw_root()
    DATASET_IDS = _discover_task1_case_ids(RAW_ROOT)
    _assign_case_splits_from_names(DATASET_IDS)

    ensure_dir(OUT_ROOT)
    ensure_dir(MERGED_ROOT)

    ACTIVE_CASE_METADATA = _prepare_case_metadata([str(c) for c in DATASET_IDS])
    ACTIVE_CONDITIONING_CHANNEL_NAMES = list(CONDITIONING_CHANNEL_NAMES) if USE_EXPLICIT_CONDITIONING else []
    print("[data] resolved RAW_ROOT:", RAW_ROOT)
    print("[data] discovered datasets:", DATASET_IDS)
    print("[data] split cases train/validation/testing:", TRAIN_CASES, VAL_CASES, TEST_CASES)
    print("[meta] validated metadata cases:", sorted(ACTIVE_CASE_METADATA.keys()))
    print("[meta] active conditioning channels:", ACTIVE_CONDITIONING_CHANNEL_NAMES)

    merged = merge_frames()

    mode = str(TASK1_TARGET_MODE).lower()
    if mode != "delta":
        raise ValueError("TASK1_TARGET_MODE must be 'delta'")
    particle_evolution_path = build_particle_evolution_dataset(merged)

    summary = {
        "raw_root": str(RAW_ROOT),
        "datasets": DATASET_IDS,
        "n_merged_frames": len(merged),
        "task1_target_mode": mode,
        "task1_particle_evolution_dataset": None if particle_evolution_path is None else str(particle_evolution_path),
        "use_explicit_conditioning": USE_EXPLICIT_CONDITIONING,
        "conditioning_channel_names": CONDITIONING_CHANNEL_NAMES if USE_EXPLICIT_CONDITIONING else [],
        "conditioning_active_channel_names": ACTIVE_CONDITIONING_CHANNEL_NAMES,
        "use_geometry_channels": USE_GEOMETRY_CHANNELS,
        "geometry_channel_names": GEOMETRY_CHANNEL_NAMES,
        "geometry_near_threshold": GEOMETRY_NEAR_THRESHOLD,
        "strict_geometry_qa": STRICT_GEOMETRY_QA,
        "geom_min_nonzero_frac": GEOM_MIN_NONZERO_FRAC,
        "geom_min_near_frac": GEOM_MIN_NEAR_FRAC,
        # "min_particles_per_frame": MIN_PARTICLES_PER_FRAME,
        "dual_split_enabled": USE_DUAL_SPLIT_PROTOCOL,
        "val_id_fraction_from_train_cases": VAL_ID_FRACTION_FROM_TRAIN_CASES,
        "val_id_min_frames_per_train_case": VAL_ID_MIN_FRAMES_PER_TRAIN_CASE,
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
