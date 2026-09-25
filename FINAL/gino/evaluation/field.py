import numpy as np
from scipy.spatial import cKDTree
from gino.data.hdf5 import read_task2_field
from .metrics import mse, relative_l2


def evaluate_native_field(predicted_xyz, predicted_velocity, hdf5_path, max_distance=None):
    """Compare native samples directly; plotting interpolation is not used here."""
    true_xyz, true_velocity = read_task2_field(hdf5_path)
    distance, index = cKDTree(predicted_xyz).query(true_xyz, k=1)
    if max_distance is not None:
        keep = distance <= float(max_distance)
        distance, index, true_velocity = distance[keep], index[keep], true_velocity[keep]
    predicted = np.asarray(predicted_velocity)[index]
    return {"mse": mse(predicted, true_velocity), "relative_l2": relative_l2(predicted, true_velocity),
            "matched_points": int(len(true_velocity)), "mean_nearest_distance": float(np.mean(distance)), "hdf5": str(hdf5_path)}
