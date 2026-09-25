from pathlib import Path
import re

import h5py
import numpy as np


def _as_xyz(value):
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr.T.astype(np.float32)
    raise ValueError(f"Expected xyz array, got {arr.shape}")


def _as_vec(value):
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr.reshape(-1, 3).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1).reshape(-1, 3).astype(np.float32)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr.T.astype(np.float32)
    raise ValueError(f"Expected vector field, got {arr.shape}")


def read_task2_field(path):
    """Read native HDF5 nodes/U and return canonical velocity plus gradients."""
    with h5py.File(path, "r") as handle:
        coords = _as_xyz(handle["nodes"])
        velocity = _as_vec(handle["U"])
    if velocity.shape[0] == 65 ** 3:
        grid = velocity.reshape(65, 65, 65, 3)
        deriv = np.gradient(grid, axis=(0, 1, 2))
        gradients = np.concatenate([part.reshape(-1, 3) for part in deriv], axis=1)
        values = np.concatenate((velocity, gradients), axis=1)
    else:
        values = velocity
    return coords, values


def find_task2_file(root, case, frame):
    case_dir = Path(root).expanduser() / str(case)
    expected = str(frame).zfill(6)
    for path in sorted(case_dir.glob("static_airfoil_fdom.*.h5")):
        numbers = re.findall(r"(\d+)(?!.*\d)", path.stem)
        if numbers and numbers[-1].zfill(6) == expected:
            return path
    return None


def native_field_plane(coords, values, y_plane=0.0, component=0):
    """Return a structured x-z slice at the nearest native y coordinate."""
    coords, values = np.asarray(coords), np.asarray(values)
    x_axis, y_axis, z_axis = (np.unique(coords[:, i]) for i in range(3))
    y_value = y_axis[int(np.argmin(np.abs(y_axis - float(y_plane))))]
    keep = np.isclose(coords[:, 1], y_value)
    plane_xyz, plane_values = coords[keep], values[keep, int(component)]
    ix = np.searchsorted(x_axis, plane_xyz[:, 0])
    iz = np.searchsorted(z_axis, plane_xyz[:, 2])
    field = np.full((len(z_axis), len(x_axis)), np.nan, dtype=np.float32)
    field[iz, ix] = plane_values
    order = np.lexsort((plane_xyz[:, 0], plane_xyz[:, 2]))
    return x_axis.astype(np.float32), z_axis.astype(np.float32), field, plane_xyz[order].astype(np.float32), plane_values[order]
