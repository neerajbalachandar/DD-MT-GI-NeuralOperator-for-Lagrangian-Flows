from pathlib import Path
import sys

import numpy as np


def particle_geometry_features(xyz, vtk_path):
    """Use the canonical VTK nearest-point rule to refresh rollout features."""
    if not vtk_path or not Path(vtk_path).is_file():
        raise FileNotFoundError(
            "Rollout needs the frame's source VTK geometry; rerun scripts/preprocess.py while the Task-1 VTK mount is available"
        )
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import process_data
    return process_data._particle_geometry_features(np.asarray(xyz), str(vtk_path), len(xyz))
