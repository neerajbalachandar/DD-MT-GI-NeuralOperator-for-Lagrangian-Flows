import numpy as np
from gino.data.hdf5 import read_task2_field
from .metrics import mse, relative_l2


def evaluate_native_field(predicted_xyz, predicted_velocity, hdf5_path):
    """Compare only predictions sampled at the exact native physical coordinates."""
    true_xyz, true_velocity = read_task2_field(hdf5_path)
    predicted_xyz, predicted_velocity = np.asarray(predicted_xyz), np.asarray(predicted_velocity)
    if predicted_xyz.shape != true_xyz.shape or not np.allclose(predicted_xyz, true_xyz, rtol=1e-6, atol=1e-7):
        raise ValueError("Native field comparison requires prediction and reference at identical coordinates")
    if predicted_velocity.shape != true_velocity.shape:
        raise ValueError(f"Field channel/shape mismatch: prediction={predicted_velocity.shape}, truth={true_velocity.shape}")
    return {"mse": mse(predicted_velocity, true_velocity), "relative_l2": relative_l2(predicted_velocity, true_velocity),
            "matched_points": int(len(true_velocity)), "comparison": "native-grid exact coordinates", "hdf5": str(hdf5_path)}
