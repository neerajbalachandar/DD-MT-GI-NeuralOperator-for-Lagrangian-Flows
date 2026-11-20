"""GPU-resident frame store for the lossless vortex-ring archives (see vortex_raw.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class CaseData:
    """All frames of one case.  Field arrays can stay on CPU for very large grids."""

    def __init__(self, path, device, load_fields=True, field_device=None):
        with np.load(path) as z:
            self.name = Path(path).stem
            self.time = torch.as_tensor(z["time"], dtype=torch.float64)
            self.dt = (self.time[1:] - self.time[:-1]).to(torch.float32)
            self.x = torch.as_tensor(z["x"], device=device)
            self.gamma = torch.as_tensor(z["gamma"], device=device)
            self.sigma = torch.as_tensor(z["sigma"], device=device)
            self.u_p = torch.as_tensor(z["particle_velocity"], device=device)
            self.J_p = torch.as_tensor(z["particle_jacobian"], device=device)
            self.valid_p = torch.as_tensor(z["particle_velocity_valid"], device=device)
            self.nodes = torch.as_tensor(z["grid_nodes"], device=device)
            self.grid_shape_zyx = tuple(int(v) for v in z["grid_shape_zyx"])
            fd = field_device or device
            if load_fields:
                self.U = torch.as_tensor(z["grid_velocity"], device=fd)
                self.W = torch.as_tensor(z["grid_vorticity"], device=fd)
            else:
                self.U = self.W = None
        self.n_frames, self.n_particles = self.x.shape[:2]

    def frame(self, t):
        return self.x[t], self.gamma[t], self.sigma[t]


def load_split(root, split, device, names=None, load_fields=True, max_grid_nodes_on_gpu=40000):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    cases = []
    for entry in manifest["cases"]:
        if entry["split"] != split or (names and entry["name"] not in names):
            continue
        n_nodes = int(np.prod(entry["grid_shape_zyx"]))
        field_device = device if n_nodes <= max_grid_nodes_on_gpu else "cpu"
        case = CaseData(root / entry["file"], device, load_fields, field_device)
        case.meta = entry
        cases.append(case)
    return cases


def ring_membership(x0: torch.Tensor, n_rings: int = 2) -> torch.Tensor:
    """Assign particles to rings from the initial axial (z) positions (k-means in 1D)."""
    z = x0[:, 2]
    centers = torch.quantile(z, torch.linspace(0.1, 0.9, n_rings, device=z.device, dtype=z.dtype))
    for _ in range(20):
        label = (z.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(1)
        centers = torch.stack([z[label == k].mean() if (label == k).any() else centers[k] for k in range(n_rings)])
    return label


def ring_diagnostics(x: torch.Tensor, gamma: torch.Tensor, label: torch.Tensor, n_rings: int):
    """Circulation-weighted axial position and radius of each ring."""
    out = []
    w = gamma.norm(dim=-1)
    rad = x[:, :2].norm(dim=-1)
    for k in range(n_rings):
        m = label == k
        if not m.any():
            out.append((float("nan"), float("nan")))
            continue
        wk = w[m] / w[m].sum()
        out.append((float((wk * x[m, 2]).sum()), float((wk * rad[m]).sum())))
    return out
