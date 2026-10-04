from pathlib import Path

import numpy as np


def particle_geometry_features(xyz, vtk_path):
    """Use the canonical VTK nearest-point rule to refresh rollout features."""
    if not vtk_path or not Path(vtk_path).is_file():
        raise FileNotFoundError(
            "Rollout needs the frame's source VTK geometry; rerun scripts/preprocess.py while the Task-1 VTK mount is available"
        )
    from .preprocessing_pipeline import _particle_geometry_features
    return _particle_geometry_features(np.asarray(xyz), str(vtk_path), len(xyz))


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
