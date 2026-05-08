"""Geometry-aware channels from VTK surface/mesh data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .vtk_loader import VTKData


@dataclass
class GeometryConfig:
    """Controls geometry feature construction such as mask thickness, SDF sign, and coordinate channels."""
    body_mask_thickness: float = 0.02
    use_signed_sdf: bool = True
    include_coordinates: bool = False


_ARRAY_ALIASES = {
    "normal": ["normal", "normals", "n", "panel_normal"],
    "surface_velocity": ["vkin", "surface_velocity", "surface_vel", "mesh_velocity", "surface_v"],
    "panel_gamma": ["panel_gamma", "gamma_panel", "gamma", "circulation_panel"],
    "freestream": ["vinf", "freestream", "uinf"],
    "induced_surface_velocity": [
        "uind",
        "induced_surface_velocity",
        "surface_induced_velocity",
        "vvpm",
        "vvpm_ab",
        "vvpm_apa",
        "vvpm_bbp",
    ],
    "lift": ["lift", "force_lift", "l"],
    "drag": ["drag", "force_drag", "d"],
    "sideforce": ["sideforce", "force_side", "side_force", "s"],
    "total_force": ["force", "total_force", "ftot"],
}


def _lower_map(d: Mapping[str, np.ndarray]) -> Dict[str, str]:
    """Create a lowercase-to-original-name lookup for case-insensitive array matching."""
    return {k.lower(): k for k in d.keys()}


def find_array(vtk_data: VTKData, aliases: Sequence[str]) -> Optional[Tuple[str, str, np.ndarray]]:
    """Find point/cell array by alias.

    Returns (association, original_name, array).
    """
    aliases_l = [a.lower() for a in aliases]

    for assoc_name, arrays in (("point", vtk_data.point_data), ("cell", vtk_data.cell_data), ("field", vtk_data.field_data)):
        lmap = _lower_map(arrays)
        for alias in aliases_l:
            for candidate_l, original in lmap.items():
                if alias == candidate_l or alias in candidate_l:
                    return assoc_name, original, arrays[original]
    return None


def _nearest_indices(src_points: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """Find nearest source-point indices for each query point using KD-tree lookup."""
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as exc:
        raise RuntimeError("scipy is required for nearest-neighbor geometry sampling") from exc

    tree = cKDTree(src_points)
    _, idx = tree.query(query_points, k=1)
    return np.asarray(idx, dtype=np.int64)


def _sample_nearest(src_points: np.ndarray, src_values: np.ndarray, grid_xyz: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sample source values at every target grid point."""
    flat = grid_xyz.reshape(-1, 3)
    idx = _nearest_indices(src_points, flat)
    values = np.asarray(src_values)
    sampled = values[idx]
    return sampled


def _cell_centers(vtk_data: VTKData) -> Optional[np.ndarray]:
    """Compute cell centers from mesh cells so cell-associated arrays can be sampled on the grid."""
    mesh = vtk_data.raw_mesh
    if mesh is not None:
        try:
            if hasattr(mesh, "cell_centers"):
                centers = mesh.cell_centers().points
                c = np.asarray(centers)
                if c.ndim == 2 and c.shape[1] == 3 and c.shape[0] > 0:
                    return c
        except Exception:
            pass

    # VTK fallback for non-pyvista raw objects.
    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    except Exception:
        return None

    raw = vtk_data.raw_mesh
    if raw is None:
        return None

    try:
        filt = vtk.vtkCellCenters()
        filt.SetInputData(raw)
        filt.Update()
        out = filt.GetOutput()
        pts = out.GetPoints()
        if pts is None:
            return None
        arr = vtk_to_numpy(pts.GetData())
        c = np.asarray(arr)
        if c.ndim == 2 and c.shape[1] == 3 and c.shape[0] > 0:
            return c
    except Exception:
        return None

    return None


def _association_points(vtk_data: VTKData, association: str) -> Optional[np.ndarray]:
    """Return the point cloud corresponding to the array association (point or cell)."""
    if association == "point":
        if vtk_data.points.ndim == 2 and vtk_data.points.shape[1] == 3 and vtk_data.points.shape[0] > 0:
            return vtk_data.points
        return None
    if association == "cell":
        return _cell_centers(vtk_data)
    return None


def _sample_alias_scalar_to_grid(vtk_data: VTKData, aliases: Sequence[str], grid_xyz: np.ndarray) -> Optional[np.ndarray]:
    """Lookup a scalar array by aliases and sample it onto the grid via nearest-neighbor."""
    match = find_array(vtk_data, aliases)
    if match is None:
        return None
    association, _, arr = match
    points = _association_points(vtk_data, association)
    if points is None:
        return None

    a = np.asarray(arr).reshape(-1)
    n = min(points.shape[0], a.shape[0])
    if n <= 0:
        return None
    sampled = _sample_nearest(points[:n], a[:n], grid_xyz).reshape(grid_xyz.shape[:3])
    return sampled.astype(np.float32)


def _sample_alias_vector_to_grid(vtk_data: VTKData, aliases: Sequence[str], grid_xyz: np.ndarray) -> Optional[np.ndarray]:
    """Lookup a vector array by aliases and sample it onto the grid via nearest-neighbor."""
    match = find_array(vtk_data, aliases)
    if match is None:
        return None
    association, _, arr = match
    points = _association_points(vtk_data, association)
    if points is None:
        return None

    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] != 3:
        return None
    n = min(points.shape[0], a.shape[0])
    if n <= 0:
        return None
    sampled = _sample_nearest(points[:n], a[:n], grid_xyz)
    return sampled.astype(np.float32)


def _extract_point_normals(vtk_data: VTKData) -> Optional[np.ndarray]:
    """Fetch surface normals from arrays or reconstruct point normals from mesh geometry."""
    match = find_array(vtk_data, _ARRAY_ALIASES["normal"])
    if match is not None:
        _, _, arr = match
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == 3:
            if a.shape[0] == vtk_data.points.shape[0]:
                return a

    mesh = vtk_data.raw_mesh
    if mesh is not None:
        try:
            # pyvista mesh path
            if hasattr(mesh, "compute_normals"):
                nmesh = mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False)
                for key in ("Normals", "normals", "Normal"):
                    if key in nmesh.point_data:
                        return np.asarray(nmesh.point_data[key])
        except Exception:
            pass

    return None


def _estimate_signed_distance(
    surface_points: np.ndarray,
    grid_flat: np.ndarray,
    normals: Optional[np.ndarray],
) -> np.ndarray:
    """Approximate signed distance using nearest-point normal orientation.

    If normals are unavailable, returns unsigned distance.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as exc:
        raise RuntimeError("scipy is required for SDF distance queries") from exc

    tree = cKDTree(surface_points)
    d, idx = tree.query(grid_flat, k=1)
    d = np.asarray(d, dtype=np.float64)

    if normals is None or normals.shape[0] != surface_points.shape[0]:
        return d

    nearest_p = surface_points[idx]
    nearest_n = normals[idx]
    rel = grid_flat - nearest_p
    # Convention: positive outside, negative inside.
    sign = np.sign(np.sum(rel * nearest_n, axis=1))
    sign[sign == 0] = 1.0
    return d * sign


def _expand_vector_to_grid(v: np.ndarray, grid_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split flattened vector values into x/y/z grids with target structured shape."""
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"Expected vector with shape (N,3), got {v.shape}")
    vx = v[:, 0].reshape(grid_shape)
    vy = v[:, 1].reshape(grid_shape)
    vz = v[:, 2].reshape(grid_shape)
    return vx, vy, vz


def _broadcast_scalar(value: float, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    """Broadcast one scalar value into a full 3D grid channel."""
    return np.full(grid_shape, float(value), dtype=np.float32)


def compute_surface_velocity(
    vtk_data: VTKData,
    grid_xyz: np.ndarray,
    prev_vtk_data: Optional[VTKData] = None,
    dt: Optional[float] = None,
) -> np.ndarray:
    """Get surface velocity sampled on target grid.

    Priority:
      1) explicit VTK arrays (e.g. Vkin)
      2) finite difference from mesh motion between frames
      3) zeros fallback
    """
    sampled = _sample_alias_vector_to_grid(vtk_data, _ARRAY_ALIASES["surface_velocity"], grid_xyz)
    if sampled is not None:
        return sampled

    if prev_vtk_data is not None and dt is not None and dt > 0:
        p_now = vtk_data.points
        p_prev = prev_vtk_data.points
        m = min(p_now.shape[0], p_prev.shape[0])
        if m > 0:
            vel = np.zeros_like(p_now, dtype=np.float64)
            vel[:m] = (p_now[:m] - p_prev[:m]) / float(dt)
            return _sample_nearest(p_now, vel, grid_xyz).astype(np.float32)

    return np.zeros((grid_xyz.reshape(-1, 3).shape[0], 3), dtype=np.float32)


def build_geometry_channels(
    grid_xyz: np.ndarray,
    vtk_data: Optional[VTKData],
    cfg: Optional[GeometryConfig] = None,
    prev_vtk_data: Optional[VTKData] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Create geometry-conditioned channels on the structured grid."""
    cfg = cfg or GeometryConfig()
    context = dict(context or {})

    grid_shape = grid_xyz.shape[:3]
    flat = grid_xyz.reshape(-1, 3)

    channels: Dict[str, np.ndarray] = {}

    if vtk_data is None or vtk_data.points.size == 0:
        # Geometry unavailable: provide deterministic zero channels for compatibility.
        zeros = np.zeros(grid_shape, dtype=np.float32)
        for key in [
            "body_mask",
            "signed_distance_field",
            "normal_x",
            "normal_y",
            "normal_z",
            "surface_velocity_x",
            "surface_velocity_y",
            "surface_velocity_z",
        ]:
            channels[key] = zeros.copy()
    else:
        points = vtk_data.points
        normals = _extract_point_normals(vtk_data)

        sdf = _estimate_signed_distance(points, flat, normals if cfg.use_signed_sdf else None)
        sdf_grid = sdf.reshape(grid_shape).astype(np.float32)
        channels["signed_distance_field"] = sdf_grid

        mask = (np.abs(sdf_grid) <= float(cfg.body_mask_thickness)).astype(np.float32)
        if cfg.use_signed_sdf:
            mask = np.maximum(mask, (sdf_grid < 0).astype(np.float32))
        channels["body_mask"] = mask

        if normals is None:
            normal_flat = np.zeros((flat.shape[0], 3), dtype=np.float64)
        else:
            normal_flat = _sample_nearest(points, normals, grid_xyz)
            nrm = np.linalg.norm(normal_flat, axis=1, keepdims=True)
            normal_flat = normal_flat / np.maximum(nrm, 1e-12)

        nx, ny, nz = _expand_vector_to_grid(normal_flat, grid_shape)
        channels["normal_x"] = nx.astype(np.float32)
        channels["normal_y"] = ny.astype(np.float32)
        channels["normal_z"] = nz.astype(np.float32)

        surf_vel_flat = compute_surface_velocity(
            vtk_data,
            grid_xyz=grid_xyz,
            prev_vtk_data=prev_vtk_data,
            dt=context.get("dt"),
        )
        if bool(context.get("stationary", False)):
            surf_vel_flat = np.zeros_like(surf_vel_flat)
        svx, svy, svz = _expand_vector_to_grid(surf_vel_flat, grid_shape)
        channels["surface_velocity_x"] = svx.astype(np.float32)
        channels["surface_velocity_y"] = svy.astype(np.float32)
        channels["surface_velocity_z"] = svz.astype(np.float32)

        # Optional panel/flow state arrays if available.
        panel_gamma = _sample_alias_scalar_to_grid(vtk_data, _ARRAY_ALIASES["panel_gamma"], grid_xyz)
        if panel_gamma is not None:
            channels["panel_gamma"] = panel_gamma

        induced = _sample_alias_vector_to_grid(vtk_data, _ARRAY_ALIASES["induced_surface_velocity"], grid_xyz)
        if induced is not None:
            ix, iy, iz = _expand_vector_to_grid(induced, grid_shape)
            channels["induced_surface_velocity_x"] = ix.astype(np.float32)
            channels["induced_surface_velocity_y"] = iy.astype(np.float32)
            channels["induced_surface_velocity_z"] = iz.astype(np.float32)

        lift = _sample_alias_vector_to_grid(vtk_data, _ARRAY_ALIASES["lift"], grid_xyz)
        if lift is not None:
            lx, ly, lz = _expand_vector_to_grid(lift, grid_shape)
            channels["lift_x"] = lx.astype(np.float32)
            channels["lift_y"] = ly.astype(np.float32)
            channels["lift_z"] = lz.astype(np.float32)

        drag = _sample_alias_vector_to_grid(vtk_data, _ARRAY_ALIASES["drag"], grid_xyz)
        if drag is not None:
            dx, dy, dz = _expand_vector_to_grid(drag, grid_shape)
            channels["drag_x"] = dx.astype(np.float32)
            channels["drag_y"] = dy.astype(np.float32)
            channels["drag_z"] = dz.astype(np.float32)

        side = _sample_alias_vector_to_grid(vtk_data, _ARRAY_ALIASES["sideforce"], grid_xyz)
        if side is not None:
            sx, sy, sz = _expand_vector_to_grid(side, grid_shape)
            channels["sideforce_x"] = sx.astype(np.float32)
            channels["sideforce_y"] = sy.astype(np.float32)
            channels["sideforce_z"] = sz.astype(np.float32)

        total = _sample_alias_vector_to_grid(vtk_data, _ARRAY_ALIASES["total_force"], grid_xyz)
        if total is not None:
            fx, fy, fz = _expand_vector_to_grid(total, grid_shape)
            channels["total_force_x"] = fx.astype(np.float32)
            channels["total_force_y"] = fy.astype(np.float32)
            channels["total_force_z"] = fz.astype(np.float32)

    # Global scalar conditioning (broadcast channels).
    phase = float(context.get("phase", 0.0))
    aoa = float(context.get("angle_of_attack", 0.0))
    reynolds = float(context.get("Reynolds_number", 0.0))
    reduced_freq = float(context.get("reduced_frequency", 0.0))

    channels["phase"] = _broadcast_scalar(phase, grid_shape)
    channels["angle_of_attack"] = _broadcast_scalar(aoa, grid_shape)
    channels["Reynolds_number"] = _broadcast_scalar(reynolds, grid_shape)
    channels["reduced_frequency"] = _broadcast_scalar(reduced_freq, grid_shape)

    freestream = context.get("freestream", None)
    if freestream is None and vtk_data is not None:
        f_vec = _sample_alias_vector_to_grid(vtk_data, _ARRAY_ALIASES["freestream"], grid_xyz)
        if f_vec is not None and f_vec.shape[0] > 0:
            fmean = np.mean(f_vec, axis=0)
            freestream = [float(fmean[0]), float(fmean[1]), float(fmean[2])]

    if freestream is not None:
        f = np.asarray(freestream).reshape(-1)
        if f.shape[0] == 3:
            channels["freestream_x"] = _broadcast_scalar(float(f[0]), grid_shape)
            channels["freestream_y"] = _broadcast_scalar(float(f[1]), grid_shape)
            channels["freestream_z"] = _broadcast_scalar(float(f[2]), grid_shape)

    if cfg.include_coordinates or bool(context.get("include_coordinates", False)):
        channels["X"] = grid_xyz[..., 0].astype(np.float32)
        channels["Y"] = grid_xyz[..., 1].astype(np.float32)
        channels["Z"] = grid_xyz[..., 2].astype(np.float32)

    return channels
