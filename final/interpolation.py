"""Field remapping utilities with prioritized interpolation fallbacks."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _as_points(arr: np.ndarray) -> np.ndarray:
    """Validate that coordinates are in `(N, 3)` form and return them as numpy array."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {a.shape}")
    return a


def _to_channels_last(values: np.ndarray) -> np.ndarray:
    """Normalize field layout to channels-last representation for interpolation backends."""
    v = np.asarray(values)
    if v.ndim == 4 and v.shape[-1] in (1, 3, 6, 9, 12):
        return v
    if v.ndim == 4 and v.shape[0] in (1, 3, 6, 9, 12):
        return np.moveaxis(v, 0, -1)
    if v.ndim == 2:
        return v[:, None] if v.shape[1] != 3 else v
    if v.ndim == 1:
        return v[:, None]
    return v


def direct_reshape_if_possible(
    source_values: np.ndarray,
    target_grid_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    """Return reshaped field if no interpolation is needed."""
    v = _to_channels_last(source_values)
    n_target = int(np.prod(target_grid_shape))

    if v.ndim == 4:
        if tuple(v.shape[:3]) == tuple(target_grid_shape):
            return v.astype(np.float32)

    if v.ndim == 2 and v.shape[0] == n_target:
        c = v.shape[1]
        return v.reshape(*target_grid_shape, c).astype(np.float32)

    return None


def _infer_structured_axes(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Detect structured-grid axes from point cloud if it lies on a Cartesian product grid."""
    p = _as_points(points)
    xs = np.unique(np.round(p[:, 0], decimals=12))
    ys = np.unique(np.round(p[:, 1], decimals=12))
    zs = np.unique(np.round(p[:, 2], decimals=12))
    if xs.size * ys.size * zs.size == p.shape[0]:
        return xs, ys, zs
    return None


def _reshape_to_structured(points: np.ndarray, values: np.ndarray):
    """Reconstruct structured value volume from scattered points when exact axes are inferred."""
    axes = _infer_structured_axes(points)
    if axes is None:
        return None

    xs, ys, zs = axes
    nx, ny, nz = xs.size, ys.size, zs.size

    p = _as_points(points)
    v = _to_channels_last(values)
    if v.ndim != 2:
        v = v.reshape(p.shape[0], -1)

    # Create lookup from coordinate tuple -> row index.
    coord_to_i = {
        (round(float(x), 12), round(float(y), 12), round(float(z), 12)): i
        for i, (x, y, z) in enumerate(p)
    }

    grid_vals = np.zeros((nx, ny, nz, v.shape[1]), dtype=np.float64)
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                i = coord_to_i.get((round(float(x), 12), round(float(y), 12), round(float(z), 12)))
                if i is None:
                    return None
                grid_vals[ix, iy, iz] = v[i]

    return axes, grid_vals


def _trilinear_interpolate(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_grid_xyz: np.ndarray,
) -> Optional[np.ndarray]:
    """Use `RegularGridInterpolator` for trilinear remapping when source grid is structured."""
    try:
        from scipy.interpolate import RegularGridInterpolator  # type: ignore
    except Exception:
        return None

    structured = _reshape_to_structured(source_points, source_values)
    if structured is None:
        return None

    (xs, ys, zs), vals = structured
    target_flat = target_grid_xyz.reshape(-1, 3)

    out = np.zeros((target_flat.shape[0], vals.shape[-1]), dtype=np.float64)
    for c in range(vals.shape[-1]):
        interp = RegularGridInterpolator(
            (xs, ys, zs),
            vals[..., c],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        out[:, c] = interp(target_flat)

    return out.reshape(*target_grid_xyz.shape[:3], vals.shape[-1]).astype(np.float32)


def _griddata_linear(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_grid_xyz: np.ndarray,
) -> Optional[np.ndarray]:
    """Use SciPy `griddata` linear interpolation with nearest fill outside convex hull."""
    try:
        from scipy.interpolate import griddata  # type: ignore
    except Exception:
        return None

    p = _as_points(source_points)
    v = _to_channels_last(source_values)
    if v.ndim != 2:
        v = v.reshape(p.shape[0], -1)

    q = target_grid_xyz.reshape(-1, 3)
    out = np.zeros((q.shape[0], v.shape[1]), dtype=np.float64)

    for c in range(v.shape[1]):
        interp = griddata(p, v[:, c], q, method="linear", fill_value=np.nan)
        if np.isnan(interp).any():
            # nearest fill for convex-hull misses.
            nearest = griddata(p, v[:, c], q, method="nearest")
            interp = np.where(np.isnan(interp), nearest, interp)
        out[:, c] = interp

    return out.reshape(*target_grid_xyz.shape[:3], v.shape[1]).astype(np.float32)


def _rbf_fallback(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_grid_xyz: np.ndarray,
) -> Optional[np.ndarray]:
    """Fallback remapping using RBF interpolation when other interpolation modes are unavailable."""
    p = _as_points(source_points)
    v = _to_channels_last(source_values)
    if v.ndim != 2:
        v = v.reshape(p.shape[0], -1)

    q = target_grid_xyz.reshape(-1, 3)

    try:
        from scipy.interpolate import RBFInterpolator  # type: ignore

        out = np.zeros((q.shape[0], v.shape[1]), dtype=np.float64)
        for c in range(v.shape[1]):
            rbf = RBFInterpolator(p, v[:, c], kernel="thin_plate_spline")
            out[:, c] = rbf(q)
        return out.reshape(*target_grid_xyz.shape[:3], v.shape[1]).astype(np.float32)

    except Exception:
        try:
            from scipy.interpolate import Rbf  # type: ignore

            out = np.zeros((q.shape[0], v.shape[1]), dtype=np.float64)
            for c in range(v.shape[1]):
                rbf = Rbf(p[:, 0], p[:, 1], p[:, 2], v[:, c], function="linear")
                out[:, c] = rbf(q[:, 0], q[:, 1], q[:, 2])
            return out.reshape(*target_grid_xyz.shape[:3], v.shape[1]).astype(np.float32)
        except Exception:
            return None


def remap_to_grid(
    source_values: np.ndarray,
    target_grid_xyz: np.ndarray,
    source_points: Optional[np.ndarray] = None,
    method_priority: Sequence[str] = ("direct", "trilinear", "griddata", "rbf"),
) -> Tuple[np.ndarray, str]:
    """Map source field values onto target structured grid.

    Priority order defaults to:
      1) direct reshape
      2) trilinear interpolation
      3) griddata linear
      4) RBF fallback
    """
    target_shape = tuple(int(s) for s in target_grid_xyz.shape[:3])

    for method in method_priority:
        m = method.lower()

        if m == "direct":
            direct = direct_reshape_if_possible(source_values, target_shape)
            if direct is not None:
                return direct, "direct"

        if source_points is None:
            continue

        if m == "trilinear":
            out = _trilinear_interpolate(source_points, source_values, target_grid_xyz)
            if out is not None:
                return out, "trilinear"

        if m == "griddata":
            out = _griddata_linear(source_points, source_values, target_grid_xyz)
            if out is not None:
                return out, "griddata_linear"

        if m == "rbf":
            out = _rbf_fallback(source_points, source_values, target_grid_xyz)
            if out is not None:
                return out, "rbf"

    raise RuntimeError(
        "Unable to remap field to target grid. Provide compatible source points/values or enable scipy backends."
    )
