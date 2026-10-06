from pathlib import Path
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _load_vtk_geom_cached(vtk_path: str):
    from .preprocessing_pipeline import _load_vtk_geom
    return _load_vtk_geom(vtk_path)


def particle_geometry_features(xyz, vtk_path):
    """Use the canonical VTK nearest-point rule to refresh rollout features."""
    if not vtk_path or not Path(vtk_path).is_file():
        raise FileNotFoundError(
            "Rollout needs the frame's source VTK geometry; rerun scripts/preprocess.py while the Task-1 VTK mount is available"
        )
    from .preprocessing_pipeline import GEOMETRY_NEAR_THRESHOLD, USE_GEOMETRY_CHANNELS

    q = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    zeros = np.zeros(len(q), dtype=np.float64)
    if not USE_GEOMETRY_CHANNELS:
        return {"geom_dist": zeros, "geom_nx": zeros.copy(), "geom_ny": zeros.copy(),
                "geom_nz": zeros.copy(), "geom_body_near": zeros.copy()}
    points, normals, tree = _load_vtk_geom_cached(str(vtk_path))
    if points is None or normals is None or len(points) == 0:
        return {"geom_dist": zeros, "geom_nx": zeros.copy(), "geom_ny": zeros.copy(),
                "geom_nz": zeros.copy(), "geom_body_near": zeros.copy()}
    if tree is not None:
        distances, indices = tree.query(q, k=1)
    else:
        diff = q[:, None, :] - points[None, :, :]
        squared_distances = np.sum(diff * diff, axis=2)
        indices = np.argmin(squared_distances, axis=1)
        distances = np.sqrt(np.min(squared_distances, axis=1))
    nearest_normals = normals[indices]
    return {
        "geom_dist": np.asarray(distances, dtype=np.float64),
        "geom_nx": np.asarray(nearest_normals[:, 0], dtype=np.float64),
        "geom_ny": np.asarray(nearest_normals[:, 1], dtype=np.float64),
        "geom_nz": np.asarray(nearest_normals[:, 2], dtype=np.float64),
        "geom_body_near": (distances <= GEOMETRY_NEAR_THRESHOLD).astype(np.float64),
    }


def estimate_chordwise_bounds(vtk_path):
    """Estimate leading/trailing x bounds from the case's airfoil VTK surface."""
    if not vtk_path or not Path(vtk_path).is_file():
        raise FileNotFoundError(
            "Cannot infer chord for z/c profiles: configure evaluation.reference_chord and "
            "evaluation.reference_leading_edge_x, or keep the source VTK geometry accessible."
        )
    from .preprocessing_pipeline import _load_vtk_geom
    points, _, _ = _load_vtk_geom(str(vtk_path))
    if points is None or np.asarray(points).ndim != 2 or np.asarray(points).shape[1] != 3:
        raise ValueError(f"Could not read airfoil points from VTK geometry {vtk_path}")
    x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    if not np.isfinite(x_min + x_max) or x_max <= x_min:
        raise ValueError(f"Invalid chordwise x bounds [{x_min}, {x_max}] in {vtk_path}")
    return x_min, x_max - x_min
