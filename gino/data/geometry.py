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
