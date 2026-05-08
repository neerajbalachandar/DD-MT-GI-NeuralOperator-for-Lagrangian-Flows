"""Build geometry-aware grid datasets and particle-surrogate datasets.

This script creates new outputs under `final/` and does not modify legacy files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geometry_features import GeometryConfig, build_geometry_channels
from .interpolation import remap_to_grid
from .io import (
    FrameEntry,
    discover_case_entries,
    ensure_dir,
    load_config,
    npz_to_dict,
    save_json,
    save_split_index,
    split_entries,
)
from .normalization import apply_channel_norm, fit_stats_from_npz, save_stats
from .projection import ProjectionConfig, ensure_scalar_particle_channels, project_particle_channels
from .vtk_loader import load_vtk, summarize_vtk_arrays


PRESET_DEFINITIONS: Dict[str, List[str] | str] = {
    "A": ["gamma_mag"],
    "B": ["Gamma_x", "Gamma_y", "Gamma_z", "sigma", "density"],
    "C": ["Gamma_x", "Gamma_y", "Gamma_z", "sigma", "density", "body_mask", "signed_distance_field"],
    "D": [
        "Gamma_x",
        "Gamma_y",
        "Gamma_z",
        "sigma",
        "density",
        "body_mask",
        "signed_distance_field",
        "X",
        "Y",
        "Z",
    ],
    "E": [
        "Gamma_x",
        "Gamma_y",
        "Gamma_z",
        "sigma",
        "density",
        "body_mask",
        "signed_distance_field",
        "X",
        "Y",
        "Z",
        "phase",
        "angle_of_attack",
        "surface_velocity_x",
        "surface_velocity_y",
        "surface_velocity_z",
    ],
    "F": "__ALL__",
}


PARTICLE_ALIASES = {
    "x": ["x", "px", "xp", "particle_x"],
    "y": ["y", "py", "yp", "particle_y"],
    "z": ["z", "pz", "zp", "particle_z"],
    "xyz": ["xyz", "points", "particles_xyz", "particle_xyz", "coords"],
    "gamma_vec": ["gamma_vec", "Gamma_vec", "Gamma", "gamma", "omega", "vorticity_particle"],
    "Gamma_x": ["Gamma_x", "gamma_x", "Gx", "gx"],
    "Gamma_y": ["Gamma_y", "gamma_y", "Gy", "gy"],
    "Gamma_z": ["Gamma_z", "gamma_z", "Gz", "gz"],
    "sigma": ["sigma", "sgm", "core_size"],
    "density": ["density", "rho_p", "particle_density"],
    "vol": ["vol", "volume", "particle_volume"],
    "circulation": ["circulation", "circ", "Gamma_scalar", "gamma_mag"],
    "static": ["static", "is_static", "static_flag"],
    "velocity": ["velocity", "U_p", "particle_velocity", "Vp"],
    "omega": ["omega", "vorticity", "W_p"],
    "gradU": ["gradU", "velocity_gradient", "grad_u", "gradU_tensor"],
}


FIELD_ALIASES = {
    "U": ["U", "velocity", "u_field", "U_grid", "flow_velocity"],
    "W": ["W", "omega", "vorticity", "w_field", "vorticity_field"],
    "points": ["points", "grid_points", "coords", "xyz", "Xgrid"],
    "X": ["X", "x_grid"],
    "Y": ["Y", "y_grid"],
    "Z": ["Z", "z_grid"],
}


@dataclass
class GridSpec:
    """Structured target grid definition (bounds + resolution) with helper to build grid coordinates."""
    bounds: Tuple[float, float, float, float, float, float]
    resolution: Tuple[int, int, int]

    def build_xyz(self) -> np.ndarray:
        """Build `(nx, ny, nz, 3)` grid coordinates from configured bounds and resolution."""
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        nx, ny, nz = self.resolution
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        zs = np.linspace(zmin, zmax, nz)
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([xg, yg, zg], axis=-1)


@dataclass
class BuildLog:
    """Mutable counters and diagnostics collected while creating one preset dataset."""
    n_entries: int = 0
    n_samples_written: int = 0
    missing_vtk: int = 0
    missing_fields: int = 0
    interpolation_counts: Dict[str, int] = None  # type: ignore
    detected_vtk_arrays: Dict[str, Dict[str, Any]] = None  # type: ignore

    def __post_init__(self):
        """Initialize mutable dict fields to empty dictionaries when dataclass defaults are `None`."""
        if self.interpolation_counts is None:
            self.interpolation_counts = {}
        if self.detected_vtk_arrays is None:
            self.detected_vtk_arrays = {}

    def count_interp(self, mode: str) -> None:
        """Increment interpolation-mode usage counters for final build diagnostics."""
        self.interpolation_counts[mode] = self.interpolation_counts.get(mode, 0) + 1


# ---------------------------
# Array/key extraction helpers
# ---------------------------

def _find_key(data: Mapping[str, np.ndarray], aliases: Sequence[str]) -> Optional[str]:
    """Find best-matching key in a dictionary using exact then substring alias matching."""
    lower_to_real = {k.lower(): k for k in data.keys()}
    for alias in aliases:
        a = alias.lower()
        if a in lower_to_real:
            return lower_to_real[a]
    for alias in aliases:
        a = alias.lower()
        for lk, rk in lower_to_real.items():
            if a in lk:
                return rk
    return None


def _extract_xyz(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Extract particle coordinates from supported key patterns and normalize to `(N, 3)`."""
    key = _find_key(data, PARTICLE_ALIASES["xyz"])
    if key is not None:
        xyz = np.asarray(data[key])
        if xyz.ndim == 2 and xyz.shape[1] == 3:
            return xyz.astype(np.float64)
        if xyz.ndim == 2 and xyz.shape[0] == 3:
            return xyz.T.astype(np.float64)

    kx = _find_key(data, PARTICLE_ALIASES["x"])
    ky = _find_key(data, PARTICLE_ALIASES["y"])
    kz = _find_key(data, PARTICLE_ALIASES["z"])
    if kx and ky and kz:
        x = np.asarray(data[kx]).reshape(-1)
        y = np.asarray(data[ky]).reshape(-1)
        z = np.asarray(data[kz]).reshape(-1)
        n = min(len(x), len(y), len(z))
        return np.stack([x[:n], y[:n], z[:n]], axis=1).astype(np.float64)

    raise KeyError("Could not locate particle coordinates (x,y,z or xyz)")


def _extract_gamma_vec(data: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    """Extract full Gamma vector channels and provide scalar-to-vector fallback if needed."""
    kvec = _find_key(data, PARTICLE_ALIASES["gamma_vec"])
    if kvec is not None:
        g = np.asarray(data[kvec])
        if g.ndim == 2 and g.shape[1] == 3:
            return g[:n].astype(np.float64)
        if g.ndim == 2 and g.shape[0] == 3:
            return g.T[:n].astype(np.float64)
        if g.ndim == 1:
            # Scalar gamma fallback: preserve by mapping to z-component.
            gz = g[:n]
            return np.stack([np.zeros_like(gz), np.zeros_like(gz), gz], axis=1).astype(np.float64)

    kx = _find_key(data, PARTICLE_ALIASES["Gamma_x"])
    ky = _find_key(data, PARTICLE_ALIASES["Gamma_y"])
    kz = _find_key(data, PARTICLE_ALIASES["Gamma_z"])
    if kx and ky and kz:
        gx = np.asarray(data[kx]).reshape(-1)[:n]
        gy = np.asarray(data[ky]).reshape(-1)[:n]
        gz = np.asarray(data[kz]).reshape(-1)[:n]
        return np.stack([gx, gy, gz], axis=1).astype(np.float64)

    raise KeyError("Could not locate Gamma vector channels (Gamma_x/y/z or Gamma_vec)")


def _extract_optional_scalar(data: Mapping[str, np.ndarray], aliases: Sequence[str], n: int, default: float = 0.0) -> np.ndarray:
    """Extract optional scalar particle channel with robust shape and default handling."""
    k = _find_key(data, aliases)
    if k is None:
        return np.full(n, default, dtype=np.float64)
    v = np.asarray(data[k]).reshape(-1)
    if v.size == 1:
        return np.full(n, float(v[0]), dtype=np.float64)
    if v.shape[0] < n:
        out = np.full(n, default, dtype=np.float64)
        out[: v.shape[0]] = v
        return out
    return v[:n].astype(np.float64)


def _extract_optional_vector(data: Mapping[str, np.ndarray], aliases: Sequence[str], n: int) -> Optional[np.ndarray]:
    """Extract optional vector particle channel in `(N, 3)` form when available."""
    k = _find_key(data, aliases)
    if k is None:
        return None
    v = np.asarray(data[k])
    if v.ndim == 2 and v.shape[1] == 3:
        return v[:n].astype(np.float64)
    if v.ndim == 2 and v.shape[0] == 3:
        return v.T[:n].astype(np.float64)
    return None


def extract_particle_channels(data: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract particle channels and preserve full Gamma vector."""
    xyz = _extract_xyz(data)
    n = xyz.shape[0]
    gamma = _extract_gamma_vec(data, n)

    sigma = _extract_optional_scalar(data, PARTICLE_ALIASES["sigma"], n, default=1e-2)
    vol = _extract_optional_scalar(data, PARTICLE_ALIASES["vol"], n, default=1.0)
    density = _extract_optional_scalar(data, PARTICLE_ALIASES["density"], n, default=1.0)
    circulation = _extract_optional_scalar(data, PARTICLE_ALIASES["circulation"], n, default=np.linalg.norm(gamma, axis=1).mean())
    static = _extract_optional_scalar(data, PARTICLE_ALIASES["static"], n, default=0.0)

    omega = _extract_optional_vector(data, PARTICLE_ALIASES["omega"], n)
    vel = _extract_optional_vector(data, PARTICLE_ALIASES["velocity"], n)

    channels: Dict[str, np.ndarray] = {
        "Gamma_x": gamma[:, 0],
        "Gamma_y": gamma[:, 1],
        "Gamma_z": gamma[:, 2],
        "gamma_mag": np.linalg.norm(gamma, axis=1),
        "sigma": sigma,
        "vol": vol,
        "density": density,
        "circulation": circulation,
        "static": static,
    }

    if omega is not None:
        channels.update(
            {
                "omega_x": omega[:, 0],
                "omega_y": omega[:, 1],
                "omega_z": omega[:, 2],
            }
        )

    if vel is not None:
        channels.update(
            {
                "particle_velocity_x": vel[:, 0],
                "particle_velocity_y": vel[:, 1],
                "particle_velocity_z": vel[:, 2],
            }
        )

    # Include any additional scalar arrays with matching particle length.
    for key, arr in data.items():
        if key in channels:
            continue
        a = np.asarray(arr)
        if a.ndim == 1 and a.shape[0] == n:
            channels[key] = a.astype(np.float64)

    return xyz, channels


def _extract_field_points(data: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    """Extract source field coordinates used for remapping target U/W fields to the grid."""
    k = _find_key(data, FIELD_ALIASES["points"])
    if k is not None:
        pts = np.asarray(data[k])
        if pts.ndim == 2 and pts.shape[1] == 3:
            return pts.astype(np.float64)

    kx = _find_key(data, FIELD_ALIASES["X"])
    ky = _find_key(data, FIELD_ALIASES["Y"])
    kz = _find_key(data, FIELD_ALIASES["Z"])
    if kx and ky and kz:
        X = np.asarray(data[kx])
        Y = np.asarray(data[ky])
        Z = np.asarray(data[kz])
        if X.shape == Y.shape == Z.shape:
            return np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1).astype(np.float64)

    return None


def _extract_vector_field(data: Mapping[str, np.ndarray], aliases: Sequence[str]) -> Optional[np.ndarray]:
    """Extract vector field array from possible aliases and normalize layout."""
    k = _find_key(data, aliases)
    if k is None:
        return None

    a = np.asarray(data[k])
    if a.ndim == 4 and a.shape[-1] == 3:
        return a.astype(np.float64)
    if a.ndim == 4 and a.shape[0] == 3:
        return np.moveaxis(a, 0, -1).astype(np.float64)
    if a.ndim == 2 and a.shape[1] == 3:
        return a.astype(np.float64)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T.astype(np.float64)

    return None


def extract_target_tensor(
    data: Mapping[str, np.ndarray],
    target_grid_xyz: np.ndarray,
    output_mode: str,
    interpolation_priority: Sequence[str],
) -> Tuple[np.ndarray, str, List[str]]:
    """Build CHW target tensor from U/W output mode and interpolation priority chain."""
    output_mode = output_mode.upper()
    points = _extract_field_points(data)

    fields = []
    names: List[str] = []
    interp_used = "direct"

    if "U" in output_mode:
        U = _extract_vector_field(data, FIELD_ALIASES["U"])
        if U is None:
            raise KeyError("Could not find target velocity field `U` in npz file")
        U_map, mode = remap_to_grid(U, target_grid_xyz, source_points=points, method_priority=interpolation_priority)
        interp_used = mode
        fields.append(U_map)
        names.extend(["U_x", "U_y", "U_z"])

    if "W" in output_mode:
        W = _extract_vector_field(data, FIELD_ALIASES["W"])
        if W is None:
            raise KeyError("Could not find target vorticity field `W` in npz file")
        W_map, mode = remap_to_grid(W, target_grid_xyz, source_points=points, method_priority=interpolation_priority)
        interp_used = mode if interp_used == "direct" else interp_used
        fields.append(W_map)
        names.extend(["W_x", "W_y", "W_z"])

    if not fields:
        raise ValueError(f"Unsupported OUTPUT_MODE={output_mode}; choose U, W, or UW")

    stacked = np.concatenate(fields, axis=-1)
    chw = np.moveaxis(stacked, -1, 0).astype(np.float32)
    return chw, interp_used, names


def _channel_stack(channels: Mapping[str, np.ndarray], channel_order: Sequence[str]) -> np.ndarray:
    """Stack selected 3D channels into a `(C, nx, ny, nz)` tensor following requested order."""
    arrs: List[np.ndarray] = []
    for name in channel_order:
        if name not in channels:
            raise KeyError(f"Channel `{name}` requested but not available")
        a = np.asarray(channels[name], dtype=np.float32)
        if a.ndim != 3:
            raise ValueError(f"Channel `{name}` must be 3D grid, got shape {a.shape}")
        arrs.append(a)
    return np.stack(arrs, axis=0)


def _resolve_preset_channels(
    preset: str,
    channels: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> List[str]:
    """Resolve channel list for a preset definition, including CUSTOM and ALL modes."""
    if preset == "CUSTOM":
        return list(config.get("custom_input_channels", []))

    definition = PRESET_DEFINITIONS.get(preset)
    if definition is None:
        raise ValueError(f"Unknown dataset preset: {preset}")

    if definition == "__ALL__":
        return sorted(channels.keys())

    selected = [name for name in definition if name in channels]
    missing = [name for name in definition if name not in channels]
    if missing:
        print(f"[warn] preset={preset} missing channels skipped: {missing}")
    return selected


def _entry_context(entry: FrameEntry, case_idx: int, n_case_entries: int, global_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    # General phase logic (no mandatory case-type labeling).
    # - If `phase` is provided in metadata, use it.
    # - Else, optionally infer normalized phase from frame index.
    # - For explicitly stationary-labeled cases, keep phase=0.
    """Build per-frame global context (phase, AoA, freestream, dt, etc.) for conditioning channels."""
    if "phase" in entry.meta:
        phase = float(entry.meta.get("phase", 0.0))
    else:
        auto_phase = bool(entry.meta.get("auto_phase_from_index", global_cfg.get("auto_phase_from_index", True)))
        if auto_phase and n_case_entries > 1:
            phase = float(case_idx) / float(max(1, n_case_entries - 1))
        else:
            phase = 0.0

    if bool(entry.meta.get("stationary", False)):
        phase = 0.0

    context = {
        "phase": phase,
        "angle_of_attack": entry.meta.get("angle_of_attack", entry.meta.get("aoa", 0.0)),
        "Reynolds_number": entry.meta.get("Reynolds_number", entry.meta.get("Re", 0.0)),
        "reduced_frequency": entry.meta.get("reduced_frequency", entry.meta.get("k", 0.0)),
        "freestream": entry.meta.get("freestream", global_cfg.get("default_freestream", [0.0, 0.0, 0.0])),
        "dt": entry.meta.get("time_step", global_cfg.get("default_dt", None)),
        "include_coordinates": bool(global_cfg.get("include_coordinates", False)),
        "stationary": bool(entry.meta.get("stationary", False)),
    }

    return context


def _compute_global_bounds(entries: Sequence[FrameEntry], padding_frac: float = 0.05, include_vtk: bool = True) -> Tuple[float, float, float, float, float, float]:
    """Compute global spatial bounds over all frames (particles and optionally VTK geometry)."""
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for e in entries:
        data = npz_to_dict(e.npz_path)
        xyz = _extract_xyz(data)
        mins = np.minimum(mins, xyz.min(axis=0))
        maxs = np.maximum(maxs, xyz.max(axis=0))

        if include_vtk and e.vtk_path is not None and e.vtk_path.exists():
            try:
                vtk = load_vtk(e.vtk_path)
                if vtk.points.size > 0:
                    mins = np.minimum(mins, vtk.points.min(axis=0))
                    maxs = np.maximum(maxs, vtk.points.max(axis=0))
            except Exception as exc:
                print(f"[warn] failed to include vtk bounds from {e.vtk_path}: {exc}")

    span = np.maximum(maxs - mins, 1e-9)
    mins = mins - padding_frac * span
    maxs = maxs + padding_frac * span

    return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2]))


def _collect_entries(cfg: Mapping[str, Any], root: Path) -> List[FrameEntry]:
    """Collect and sort all frame entries across configured dataset cases."""
    entries: List[FrameEntry] = []
    for case_cfg in cfg.get("cases", []):
        case_entries = discover_case_entries(case_cfg, root=root)
        entries.extend(case_entries)
    return sorted(entries, key=lambda e: (e.case_name, e.frame_id))


def _save_sample(
    out_path: Path,
    input_tensor: np.ndarray,
    target_tensor: np.ndarray,
    input_channels: Sequence[str],
    target_channels: Sequence[str],
    meta: Mapping[str, Any],
) -> None:
    """Persist one processed training sample with tensors, channel names, and metadata."""
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        input=input_tensor.astype(np.float32),
        target=target_tensor.astype(np.float32),
        input_channels=np.asarray(list(input_channels), dtype=object),
        target_channels=np.asarray(list(target_channels), dtype=object),
        meta=np.asarray(dict(meta), dtype=object),
    )


def build_field_datasets(config: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    """Main stage-2 builder for structured field-learning datasets and preset variants A-F."""
    out_root = ensure_dir(root / config.get("output_root", "final/output"))
    dataset_name = str(config.get("dataset_name", "geometry_aware"))
    dataset_root = ensure_dir(out_root / dataset_name)

    entries = _collect_entries(config, root=root)
    if not entries:
        raise RuntimeError("No entries found. Check `cases` globs in config.")

    split_cfg = config.get("split", {})
    split = split_entries(
        entries,
        train_frac=float(split_cfg.get("train", 0.7)),
        val_frac=float(split_cfg.get("val", 0.15)),
        test_frac=float(split_cfg.get("test", 0.15)),
        seed=int(split_cfg.get("seed", 42)),
        strategy=str(split_cfg.get("strategy", "random")),
    )

    save_split_index(dataset_root / "split_index.json", split)

    grid_cfg = config.get("grid", {})
    bounds_mode = str(grid_cfg.get("bounds_mode", "global"))
    if bounds_mode == "global":
        bounds = _compute_global_bounds(
            entries,
            padding_frac=float(grid_cfg.get("padding_frac", 0.05)),
            include_vtk=bool(grid_cfg.get("include_vtk_bounds", True)),
        )
    else:
        b = grid_cfg.get("bounds")
        if b is None or len(b) != 6:
            raise ValueError("grid.bounds must have 6 numbers when bounds_mode != global")
        bounds = tuple(float(x) for x in b)

    res = tuple(int(v) for v in grid_cfg.get("resolution", [32, 32, 32]))
    grid = GridSpec(bounds=bounds, resolution=res)
    grid_xyz = grid.build_xyz()

    grid_log = {
        "bounds": bounds,
        "resolution": res,
    }
    save_json(dataset_root / "grid_spec.json", grid_log)

    proj_cfg = ProjectionConfig(**config.get("projection", {}))
    geom_cfg = GeometryConfig(**config.get("geometry", {}))
    interpolation_priority = config.get("interpolation", {}).get(
        "method_priority", ["direct", "trilinear", "griddata", "rbf"]
    )

    presets = list(config.get("dataset_presets", ["A", "B", "C", "D", "E", "F"]))
    logs: Dict[str, Any] = {}

    # Case-aware ordering for motion features.
    case_to_entries: Dict[str, List[FrameEntry]] = {}
    for e in entries:
        case_to_entries.setdefault(e.case_name, []).append(e)

    entry_lookup = {(e.case_name, e.frame_id): e for e in entries}
    split_ids = {
        split_name: {(e.case_name, e.frame_id) for e in split_entries_}
        for split_name, split_entries_ in split.items()
    }

    for preset in presets:
        print(f"[build] preset={preset}")
        preset_root = ensure_dir(dataset_root / f"preset_{preset}")
        samples_dir = ensure_dir(preset_root / "samples")

        blog = BuildLog(n_entries=len(entries))
        sample_paths: Dict[Tuple[str, str], Path] = {}
        channels_used: Optional[List[str]] = None
        input_shape: Optional[Tuple[int, ...]] = None
        target_shape: Optional[Tuple[int, ...]] = None

        for case_name, case_entries in case_to_entries.items():
            prev_vtk = None
            for i, entry in enumerate(case_entries):
                context = _entry_context(entry, i, len(case_entries), config)
                data = npz_to_dict(entry.npz_path)

                xyz, pchannels = extract_particle_channels(data)
                scalar_channels = ensure_scalar_particle_channels(pchannels)

                sigma = scalar_channels.get("sigma")
                projected = project_particle_channels(
                    particle_xyz=xyz,
                    particle_channels=scalar_channels,
                    grid_xyz=grid_xyz,
                    sigma=sigma,
                    cfg=proj_cfg,
                )

                vtk_data = None
                if entry.vtk_path is not None and entry.vtk_path.exists():
                    try:
                        vtk_data = load_vtk(entry.vtk_path)
                        blog.detected_vtk_arrays[str(entry.vtk_path)] = summarize_vtk_arrays(vtk_data)
                    except Exception as exc:
                        print(f"[warn] failed to load VTK `{entry.vtk_path}`: {exc}")
                        blog.missing_vtk += 1
                else:
                    blog.missing_vtk += 1

                geom = build_geometry_channels(
                    grid_xyz=grid_xyz,
                    vtk_data=vtk_data,
                    cfg=geom_cfg,
                    prev_vtk_data=prev_vtk,
                    context=context,
                )

                all_channels = {**projected, **geom}
                channel_order = _resolve_preset_channels(preset, all_channels, config)
                if not channel_order:
                    raise RuntimeError(f"No channels selected for preset {preset}")

                x_tensor = _channel_stack(all_channels, channel_order)
                if channels_used is None:
                    channels_used = list(channel_order)
                    input_shape = tuple(int(s) for s in x_tensor.shape)

                try:
                    y_tensor, interp_used, target_names = extract_target_tensor(
                        data,
                        target_grid_xyz=grid_xyz,
                        output_mode=str(config.get("output_mode", "U")),
                        interpolation_priority=interpolation_priority,
                    )
                    blog.count_interp(interp_used)
                    if target_shape is None:
                        target_shape = tuple(int(s) for s in y_tensor.shape)
                except Exception as exc:
                    blog.missing_fields += 1
                    print(f"[warn] skipping frame due to target extraction failure: {entry.npz_path} ({exc})")
                    prev_vtk = vtk_data
                    continue

                sample_name = f"{entry.case_name}__{entry.frame_id}.npz"
                sample_path = samples_dir / sample_name

                _save_sample(
                    sample_path,
                    input_tensor=x_tensor,
                    target_tensor=y_tensor,
                    input_channels=channel_order,
                    target_channels=target_names,
                    meta={
                        "case_name": entry.case_name,
                        "frame_id": entry.frame_id,
                        "npz_path": str(entry.npz_path),
                        "vtk_path": None if entry.vtk_path is None else str(entry.vtk_path),
                        "interpolation": interp_used,
                        "context": context,
                    },
                )

                sample_paths[(entry.case_name, entry.frame_id)] = sample_path
                blog.n_samples_written += 1
                prev_vtk = vtk_data

        # Split lists for this preset (only existing written samples).
        split_files: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        for split_name, ids in split_ids.items():
            for case_frame in sorted(ids):
                path = sample_paths.get(case_frame)
                if path is not None:
                    split_files[split_name].append(str(path.relative_to(preset_root)))

        save_json(preset_root / "split_files.json", split_files)

        # Dataset-level normalization on train split only.
        norm_cfg = config.get("normalization", {})
        if bool(norm_cfg.get("enabled", True)) and split_files["train"]:
            train_abs = [preset_root / rel for rel in split_files["train"]]
            in_stats = fit_stats_from_npz(train_abs, key="input")
            out_stats = fit_stats_from_npz(train_abs, key="target")
            save_stats(preset_root / "normalization_stats.npz", in_stats, out_stats)

            if bool(norm_cfg.get("write_normalized_samples", True)):
                for rel in split_files["train"] + split_files["val"] + split_files["test"]:
                    p = preset_root / rel
                    with np.load(p, allow_pickle=True) as d:
                        x = np.asarray(d["input"], dtype=np.float32)
                        y = np.asarray(d["target"], dtype=np.float32)
                        x_norm = apply_channel_norm(x, in_stats)
                        y_norm = apply_channel_norm(y, out_stats)

                        payload = {k: d[k] for k in d.files if k not in {"input_norm", "target_norm"}}
                    payload["input_norm"] = x_norm.astype(np.float32)
                    payload["target_norm"] = y_norm.astype(np.float32)
                    np.savez_compressed(p, **payload)

        logs[preset] = {
            "n_entries": blog.n_entries,
            "n_samples_written": blog.n_samples_written,
            "missing_vtk": blog.missing_vtk,
            "missing_fields": blog.missing_fields,
            "interpolation_counts": blog.interpolation_counts,
            "channels_preset": preset,
            "channels_used": [] if channels_used is None else channels_used,
            "input_tensor_shape": input_shape,
            "target_tensor_shape": target_shape,
        }

        save_json(preset_root / "build_log.json", logs[preset])
        if blog.detected_vtk_arrays:
            save_json(preset_root / "detected_vtk_arrays.json", blog.detected_vtk_arrays)

    save_json(dataset_root / "build_summary.json", logs)

    return {
        "dataset_root": str(dataset_root),
        "grid": grid_log,
        "presets": presets,
        "logs": logs,
    }


def _extract_particle_targets(data: Mapping[str, np.ndarray], n: int) -> Optional[np.ndarray]:
    """Extract particle-level targets `(velocity, gradU)` as a 12-channel regression vector."""
    vel = _extract_optional_vector(data, PARTICLE_ALIASES["velocity"], n)
    if vel is None:
        return None

    grad = None
    kg = _find_key(data, PARTICLE_ALIASES["gradU"])
    if kg is not None:
        g = np.asarray(data[kg])
        if g.ndim == 3 and g.shape[1:] == (3, 3):
            grad = g[:n].reshape(n, 9)
        elif g.ndim == 2 and g.shape[1] == 9:
            grad = g[:n]

    # component fallback
    if grad is None:
        comps = []
        for row in ("x", "y", "z"):
            for col in ("x", "y", "z"):
                aliases = [f"gradU{row}_{col}", f"gradU{row}{col}", f"gradU_{row}{col}"]
                k = _find_key(data, aliases)
                if k is None:
                    comps = []
                    break
                comps.append(np.asarray(data[k]).reshape(-1)[:n])
            if not comps:
                break
        if comps:
            grad = np.stack(comps, axis=1)

    if grad is None:
        return None

    y = np.concatenate([vel, grad], axis=1)
    if y.shape[1] != 12:
        return None
    return y.astype(np.float32)


def build_particle_dataset(config: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    """Build particle-surrogate dataset (`inputs_particle`, `targets_particle`) for GNO training."""
    out_root = ensure_dir(root / config.get("output_root", "final/output"))
    dataset_name = str(config.get("dataset_name", "geometry_aware"))
    p_root = ensure_dir(out_root / dataset_name / "particle_dataset")

    entries = _collect_entries(config, root=root)
    if not entries:
        raise RuntimeError("No entries found for particle dataset")

    feature_names = config.get(
        "particle_input_features",
        [
            "x",
            "y",
            "z",
            "Gamma_x",
            "Gamma_y",
            "Gamma_z",
            "sigma",
            "vol",
            "circulation",
            "static",
            "angle_of_attack",
            "phase",
            "freestream_x",
            "freestream_y",
            "freestream_z",
        ],
    )

    target_names = [
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "gradUx_x",
        "gradUx_y",
        "gradUx_z",
        "gradUy_x",
        "gradUy_y",
        "gradUy_z",
        "gradUz_x",
        "gradUz_y",
        "gradUz_z",
    ]

    rows_x: List[np.ndarray] = []
    rows_y: List[np.ndarray] = []
    frame_offsets: List[Tuple[str, str, int, int]] = []
    frame_contexts: List[Dict[str, Any]] = []

    case_to_entries: Dict[str, List[FrameEntry]] = {}
    for e in entries:
        case_to_entries.setdefault(e.case_name, []).append(e)

    for case_name, case_entries in case_to_entries.items():
        for i, entry in enumerate(case_entries):
            data = npz_to_dict(entry.npz_path)
            xyz, channels = extract_particle_channels(data)
            n = xyz.shape[0]
            y = _extract_particle_targets(data, n=n)
            if y is None:
                print(f"[warn] particle targets missing in {entry.npz_path}; skipping")
                continue

            context = _entry_context(entry, i, len(case_entries), config)
            chans = dict(channels)
            chans["x"] = xyz[:, 0]
            chans["y"] = xyz[:, 1]
            chans["z"] = xyz[:, 2]
            chans["angle_of_attack"] = np.full(n, float(context.get("angle_of_attack", 0.0)))
            chans["phase"] = np.full(n, float(context.get("phase", 0.0)))
            fs = np.asarray(context.get("freestream", [0.0, 0.0, 0.0])).reshape(-1)
            if fs.shape[0] >= 3:
                chans["freestream_x"] = np.full(n, float(fs[0]))
                chans["freestream_y"] = np.full(n, float(fs[1]))
                chans["freestream_z"] = np.full(n, float(fs[2]))

            miss = [f for f in feature_names if f not in chans]
            if miss:
                for m in miss:
                    chans[m] = np.zeros(n, dtype=np.float64)

            x = np.stack([np.asarray(chans[f], dtype=np.float32) for f in feature_names], axis=1)

            start = 0 if not rows_x else int(sum(r.shape[0] for r in rows_x))
            end = start + n
            frame_offsets.append((entry.case_name, entry.frame_id, start, end))
            frame_contexts.append(
                {
                    "case_name": entry.case_name,
                    "frame_id": entry.frame_id,
                    "start": start,
                    "end": end,
                    "dt": None if context.get("dt", None) is None else float(context["dt"]),
                    "phase": float(context.get("phase", 0.0)),
                    "angle_of_attack": float(context.get("angle_of_attack", 0.0)),
                    "freestream": [float(v) for v in np.asarray(context.get("freestream", [0.0, 0.0, 0.0])).reshape(-1)[:3]],
                    "stationary": bool(context.get("stationary", False)),
                }
            )

            rows_x.append(x)
            rows_y.append(y)

    if not rows_x:
        raise RuntimeError("No particle rows assembled. Check particle velocity/gradient keys in source npz.")

    X = np.concatenate(rows_x, axis=0).astype(np.float32)
    Y = np.concatenate(rows_y, axis=0).astype(np.float32)

    np.savez_compressed(
        p_root / "particle_dataset.npz",
        inputs_particle=X,
        targets_particle=Y,
        feature_names=np.asarray(feature_names, dtype=object),
        target_names=np.asarray(target_names, dtype=object),
        frame_offsets=np.asarray(frame_offsets, dtype=object),
        frame_contexts=np.asarray(frame_contexts, dtype=object),
    )

    # row-level split generated from frame split.
    split_cfg = config.get("split", {})
    split = split_entries(
        entries,
        train_frac=float(split_cfg.get("train", 0.7)),
        val_frac=float(split_cfg.get("val", 0.15)),
        test_frac=float(split_cfg.get("test", 0.15)),
        seed=int(split_cfg.get("seed", 42)),
        strategy=str(split_cfg.get("strategy", "random")),
    )
    split_ids = {
        split_name: {(e.case_name, e.frame_id) for e in split_entries_}
        for split_name, split_entries_ in split.items()
    }

    idx = {"train": [], "val": [], "test": []}
    for case_name, frame_id, start, end in frame_offsets:
        key = (case_name, frame_id)
        for split_name in ("train", "val", "test"):
            if key in split_ids[split_name]:
                idx[split_name].append(np.arange(start, end, dtype=np.int64))

    split_arrays = {}
    for split_name, parts in idx.items():
        split_arrays[f"{split_name}_idx"] = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.int64)

    np.savez_compressed(p_root / "particle_split_indices.npz", **split_arrays)

    summary = {
        "dataset_path": str(p_root / "particle_dataset.npz"),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets": int(Y.shape[1]),
        "particle_task_mode": str(config.get("particle_task_mode", "quantities_then_physics")),
        "split_counts": {k: int(v.shape[0]) for k, v in split_arrays.items()},
    }
    save_json(p_root / "build_log.json", summary)
    return summary


def main() -> None:
    """CLI wrapper for dataset building (field datasets and/or particle dataset)."""
    parser = argparse.ArgumentParser(description="Build geometry-aware vortex-particle datasets")
    parser.add_argument("--config", type=str, default="final/configs/pipeline_config.yaml")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--build-field", action="store_true", help="Build structured field datasets")
    parser.add_argument("--build-particle", action="store_true", help="Build particle-surrogate dataset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(args.root)

    build_field = args.build_field or (not args.build_field and not args.build_particle)
    build_particle = args.build_particle or bool(cfg.get("build_particle_dataset", True))

    if build_field:
        field_summary = build_field_datasets(cfg, root=root)
        print("[done] field dataset summary")
        print(json.dumps(field_summary, indent=2))

    if build_particle:
        particle_summary = build_particle_dataset(cfg, root=root)
        print("[done] particle dataset summary")
        print(json.dumps(particle_summary, indent=2))


if __name__ == "__main__":
    main()
