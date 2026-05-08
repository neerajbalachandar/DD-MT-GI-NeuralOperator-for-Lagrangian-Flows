"""Particle-to-grid projection kernels.

Projection here is a feature encoding step, not a Biot-Savart reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


@dataclass
class ProjectionConfig:
    """Configuration for particle-to-grid deposition kernel choice, radius policy, and normalization."""
    kernel: str = "gaussian"  # "gaussian" | "compact"
    radius: float = 0.05
    support_multiplier: float = 3.0
    radius_from_sigma: bool = True
    sigma_scale: float = 1.0
    normalize_weights: bool = True
    min_weight: float = 1e-12


def _gaussian_kernel(d: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute isotropic Gaussian deposition weights from particle-to-grid distances."""
    return np.exp(-0.5 * (d / np.maximum(h, 1e-12)) ** 2)


def _wendland_c2_kernel(d: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute compact-support Wendland C2 weights for local particle deposition."""
    q = d / np.maximum(h, 1e-12)
    w = np.zeros_like(q)
    m = q < 1.0
    qm = q[m]
    w[m] = (1.0 - qm) ** 4 * (4.0 * qm + 1.0)
    return w


def _build_grid_index(grid_xyz: np.ndarray):
    """Build a KD-tree over flattened grid points for fast neighborhood queries per particle."""
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "scipy is required for projection KD-tree neighbor search."
        ) from exc

    flat = grid_xyz.reshape(-1, 3)
    return cKDTree(flat), flat


def _resolve_radius(base_radius: float, sigma_value: float, cfg: ProjectionConfig) -> float:
    """Resolve per-particle deposition radius using sigma scaling or fixed fallback radius."""
    if cfg.radius_from_sigma and np.isfinite(sigma_value) and sigma_value > 0:
        return float(cfg.sigma_scale * sigma_value)
    return float(base_radius)


def project_particle_channels(
    particle_xyz: np.ndarray,
    particle_channels: Mapping[str, np.ndarray],
    grid_xyz: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    cfg: Optional[ProjectionConfig] = None,
) -> Dict[str, np.ndarray]:
    """Project per-particle scalar channels onto structured grid.

    Parameters
    ----------
    particle_xyz
        Particle coordinates with shape (N, 3).
    particle_channels
        Dict of scalar channels, each shape (N,).
    grid_xyz
        Structured grid coordinates with shape (nx, ny, nz, 3).
    sigma
        Optional per-particle smoothing radius input.
    cfg
        Projection behavior.
    """
    cfg = cfg or ProjectionConfig()

    xyz = np.asarray(particle_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"particle_xyz must be (N,3), got {xyz.shape}")

    n = xyz.shape[0]
    if sigma is None:
        sigma_arr = np.full(n, cfg.radius, dtype=np.float64)
    else:
        sigma_arr = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if sigma_arr.shape[0] != n:
            raise ValueError("sigma length does not match particle count")

    tree, grid_flat = _build_grid_index(grid_xyz)
    m = grid_flat.shape[0]

    accum: Dict[str, np.ndarray] = {
        k: np.zeros(m, dtype=np.float64) for k in particle_channels.keys()
    }
    weight_sum = np.zeros(m, dtype=np.float64)

    kernel = cfg.kernel.lower()
    if kernel not in {"gaussian", "compact"}:
        raise ValueError(f"Unknown kernel: {cfg.kernel}")

    for i in range(n):
        h = _resolve_radius(cfg.radius, sigma_arr[i], cfg)
        support_r = h if kernel == "compact" else cfg.support_multiplier * h
        if support_r <= 0:
            continue

        nn = tree.query_ball_point(xyz[i], r=float(support_r))
        if len(nn) == 0:
            continue

        nn_idx = np.asarray(nn, dtype=np.int64)
        pts = grid_flat[nn_idx]
        d = np.linalg.norm(pts - xyz[i], axis=1)

        if kernel == "compact":
            w = _wendland_c2_kernel(d, np.full_like(d, h))
        else:
            w = _gaussian_kernel(d, np.full_like(d, h))

        if np.all(w <= 0):
            continue

        for name, values in particle_channels.items():
            v = np.asarray(values).reshape(-1)
            if v.shape[0] != n:
                raise ValueError(f"Channel `{name}` size mismatch: {v.shape[0]} != {n}")
            accum[name][nn_idx] += w * float(v[i])

        weight_sum[nn_idx] += w

    out: Dict[str, np.ndarray] = {}
    grid_shape = grid_xyz.shape[:3]

    for name, a in accum.items():
        if cfg.normalize_weights:
            norm = np.maximum(weight_sum, cfg.min_weight)
            arr = a / norm
        else:
            arr = a
        out[name] = arr.reshape(grid_shape).astype(np.float32)

    # Extra helper channel: projected particle support density.
    if cfg.normalize_weights:
        density = weight_sum / np.maximum(np.max(weight_sum), cfg.min_weight)
    else:
        density = weight_sum
    out["particle_density"] = density.reshape(grid_shape).astype(np.float32)

    return out


def split_vector_channels(name: str, arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Split vector channel into scalar components preserving names."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Expected (N,3) for vector split, got {a.shape}")

    return {
        f"{name}_x": a[:, 0],
        f"{name}_y": a[:, 1],
        f"{name}_z": a[:, 2],
    }


def ensure_scalar_particle_channels(channels: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert mixed scalar/vector particle channels into scalar-only channel dictionary."""
    out: Dict[str, np.ndarray] = {}
    for name, arr in channels.items():
        a = np.asarray(arr)
        if a.ndim == 1:
            out[name] = a
        elif a.ndim == 2 and a.shape[1] == 1:
            out[name] = a[:, 0]
        elif a.ndim == 2 and a.shape[1] == 3:
            out.update(split_vector_channels(name, a))
        else:
            # Keep first component as a deterministic fallback.
            out[name] = a.reshape(a.shape[0], -1)[:, 0]
    return out
