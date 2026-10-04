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


def physical_field_values(coords, velocity, shape=(65, 65, 65)):
    """Attach xyz derivatives, independent of the native HDF5 node row order."""
    coords, velocity = np.asarray(coords), np.asarray(velocity)
    if coords.ndim != 2 or coords.shape[1] != 3 or velocity.shape != coords.shape:
        raise ValueError(f"Expected matching [N,3] coordinates and velocity, got {coords.shape}, {velocity.shape}")
    expected_count = int(np.prod(shape)) if shape is not None else len(coords)
    if len(coords) != expected_count or len(velocity) != expected_count:
        raise ValueError(
            f"Expected N_nodes=N_U=prod(shape)={expected_count}, "
            f"got N_nodes={len(coords)}, N_U={len(velocity)}"
        )
    axes = tuple(np.unique(coords[:, axis]) for axis in range(3))
    grid_shape = tuple(len(axis) for axis in axes)
    if shape is not None and tuple(shape) != grid_shape:
        raise ValueError(f"Native grid shape {grid_shape} does not match requested shape {tuple(shape)}")
    if int(np.prod(grid_shape)) != len(coords):
        raise ValueError("Native nodes do not form a complete rectilinear xyz grid")
    indices = tuple(np.searchsorted(axes[axis], coords[:, axis]) for axis in range(3))
    flat = np.ravel_multi_index(indices, grid_shape)
    if len(np.unique(flat)) != len(coords):
        raise ValueError("Native nodes contain duplicate or missing rectilinear grid locations")

    velocity_grid = np.empty((*grid_shape, 3), dtype=np.result_type(velocity.dtype, np.float64))
    velocity_grid[indices] = velocity
    sample_rows = np.unique(np.asarray([0, len(coords) // 2, len(coords) - 1], dtype=np.int64))
    sampled_indices = tuple(index[sample_rows] for index in indices)
    if not np.array_equal(velocity_grid[sampled_indices], velocity[sample_rows]):
        raise ValueError("Native coordinate-to-velocity row pairing failed round-trip checks")
    sampled_grid_xyz = np.column_stack([axes[axis][indices[axis][sample_rows]] for axis in range(3)])
    if not np.array_equal(sampled_grid_xyz, coords[sample_rows]):
        raise ValueError("Native node coordinates failed grid-index round-trip checks")
    derivatives = np.gradient(velocity_grid, *axes, axis=(0, 1, 2), edge_order=2)
    gradient_grid = np.concatenate(derivatives, axis=-1)
    gradient_rows = gradient_grid[indices]
    return np.concatenate((velocity, gradient_rows), axis=1).astype(np.float32)


def read_task2_velocity(path):
    """Read the simulation's nodal coordinates and stored velocity without interpolation."""
    with h5py.File(path, "r") as handle:
        coords = _as_xyz(handle["nodes"])
        velocity = _as_vec(handle["U"])
    expected = 65 ** 3
    if len(coords) != expected or len(velocity) != expected:
        raise ValueError(
            f"{path}: expected N_nodes=N_U={expected}, got {len(coords)} and {len(velocity)}"
        )
    return coords, velocity


def read_task2_field(path):
    """Read native velocity and attach finite-difference gradient channels."""
    coords, velocity = read_task2_velocity(path)
    return coords, physical_field_values(coords, velocity)


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


def highest_energy_y_plane(coords, velocity):
    """Choose a native y slice with the greatest mean kinetic-energy density."""
    coords, velocity = np.asarray(coords), np.asarray(velocity)
    if coords.shape != velocity.shape or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Expected matching [N,3] coordinates and velocities")
    y_values = np.unique(coords[:, 1])
    energy = np.asarray([np.mean(np.sum(velocity[np.isclose(coords[:, 1], y)] ** 2, axis=1))
                         for y in y_values])
    return float(y_values[int(np.argmax(energy))])
