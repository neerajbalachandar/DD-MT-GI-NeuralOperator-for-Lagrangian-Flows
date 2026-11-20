"""Tests for the lossless vortex-ring archives (processed_data/vortex_rings_v2) and frame utilities."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gino.data.vortex_frames import CaseData, load_split, ring_diagnostics, ring_membership

ROOT = Path(__file__).resolve().parents[1] / "processed_data" / "vortex_rings_v2"
needs_data = pytest.mark.skipif(not (ROOT / "manifest.json").is_file(),
                                reason="processed_data/vortex_rings_v2 not built")


def _manifest():
    return json.loads((ROOT / "manifest.json").read_text())


def _train_leapfrog_entry():
    entries = [c for c in _manifest()["cases"] if c["split"] == "train"]
    leap = [c for c in entries if c["name"].startswith("leapfrog")]
    return (leap or entries)[0]


def _biot_savart_winckelmans(q, x, gamma, sigma):
    """u(q) = -1/(4 pi) sum_p (r^2 + 2.5 s^2)/(r^2 + s^2)^{5/2} (r x Gamma_p),  r = q - x_p."""
    r = q[:, None, :] - x[None, :, :]
    r2 = (r * r).sum(-1)
    s2 = (sigma ** 2)[None, :]
    f = (r2 + 2.5 * s2) / (r2 + s2) ** 2.5
    rxg = torch.cross(r, gamma[None].expand_as(r), dim=-1)
    return -(f.unsqueeze(-1) * rxg).sum(1) / (4.0 * math.pi)


# --------------------------------------------------------------------------- archives
@needs_data
def test_manifest_splits():
    cases = _manifest()["cases"]
    assert len(cases) == 19
    counts = {s: sum(c["split"] == s for c in cases) for s in ("train", "val", "test")}
    assert counts == {"train": 8, "val": 4, "test": 7}
    for c in cases:
        assert (ROOT / c["file"]).is_file(), c["file"]


@needs_data
def test_train_archive_contract():
    entry = _train_leapfrog_entry()
    with np.load(ROOT / entry["file"]) as z:
        valid = z["particle_velocity_valid"]
        F = len(z["time"])
        x, gamma, sigma = z["x"], z["gamma"], z["sigma"]
        up, Jp = z["particle_velocity"], z["particle_jacobian"]
        time, steps = z["time"], z["steps"]
        assert time.dtype == np.float64
        assert valid.shape == (F,) and valid.dtype == np.bool_
        assert not valid[0] and valid[1:].all()
        assert np.abs(up[0]).max() == 0  # FLOWVPM writes zeros at frame 0
        N = entry["particles"]
        assert x.shape == (F, N, 3) and gamma.shape == (F, N, 3) and sigma.shape == (F, N)
        assert up.shape == (F, N, 3) and Jp.shape == (F, N, 3, 3)
        assert F == entry["frames"] and len(steps) == F and np.all(np.diff(steps) > 0)
        dt = np.diff(time)
        assert np.all(dt > 0)
        np.testing.assert_allclose(dt.mean(), entry["dt_mean"], rtol=1e-9)
        np.testing.assert_allclose(dt, dt.mean(), rtol=1e-6)
        G = int(np.prod(z["grid_shape_zyx"]))
        assert z["grid_nodes"].shape == (G, 3)
        assert z["grid_velocity"].shape == (F, G, 3)


@needs_data
def test_casedata_loads_on_cpu():
    entry = _train_leapfrog_entry()
    case = CaseData(ROOT / entry["file"], "cpu", load_fields=False)
    assert case.n_frames == entry["frames"] and case.n_particles == entry["particles"]
    assert case.U is None and case.W is None
    with np.load(ROOT / entry["file"]) as z:
        np.testing.assert_allclose(case.dt.numpy(), np.diff(z["time"]).astype(np.float32), rtol=1e-6)
    x, g, s = case.frame(3)
    assert x.shape == (case.n_particles, 3) and g.shape == x.shape and s.shape == (case.n_particles,)
    cases = load_split(ROOT, "val", "cpu", load_fields=False)
    assert len(cases) == 4 and all(c.meta["split"] == "val" for c in cases)


@needs_data
def test_grid_nodes_x_fastest():
    entry = _train_leapfrog_entry()
    with np.load(ROOT / entry["file"]) as z:
        nodes, shape = z["grid_nodes"], tuple(int(v) for v in z["grid_shape_zyx"])
    nz, ny, nx = shape
    g = nodes.reshape(nz, ny, nx, 3)
    assert np.all(np.diff(g[..., 0], axis=2) > 0)            # x increases along the fastest axis
    assert np.allclose(np.diff(g[..., 0], axis=1), 0) and np.allclose(np.diff(g[..., 0], axis=0), 0)
    assert np.all(np.diff(g[..., 1], axis=1) > 0)
    assert np.allclose(np.diff(g[..., 1], axis=2), 0) and np.allclose(np.diff(g[..., 1], axis=0), 0)
    assert np.all(np.diff(g[..., 2], axis=0) > 0)
    assert np.allclose(np.diff(g[..., 2], axis=2), 0) and np.allclose(np.diff(g[..., 2], axis=1), 0)


@needs_data
def test_particle_velocity_and_jacobian_match_biot_savart():
    entry = _train_leapfrog_entry()
    with np.load(ROOT / entry["file"]) as z:
        t = len(z["time"]) // 2
        assert z["particle_velocity_valid"][t]
        x = torch.as_tensor(z["x"][t], dtype=torch.float64)
        gamma = torch.as_tensor(z["gamma"][t], dtype=torch.float64)
        sigma = torch.as_tensor(z["sigma"][t], dtype=torch.float64)
        up = torch.as_tensor(z["particle_velocity"][t], dtype=torch.float64)
        Jp = torch.as_tensor(z["particle_jacobian"][t], dtype=torch.float64)
    idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:200]
    q = x[idx].clone().requires_grad_(True)
    u = _biot_savart_winckelmans(q, x, gamma, sigma)  # sum over ALL particles incl. self pair
    rel_u = float((u.detach() - up[idx]).norm() / up[idx].norm())
    assert rel_u < 0.01, f"particle velocity vs Biot-Savart relL2 = {rel_u:.4f}"
    # J[n, i, j] = du_i / dq_j
    J = torch.stack([torch.autograd.grad(u[:, i].sum(), q, retain_graph=True)[0] for i in range(3)], 1)
    rel_J = float((J - Jp[idx]).norm() / Jp[idx].norm())
    rel_JT = float((J.transpose(1, 2) - Jp[idx]).norm() / Jp[idx].norm())
    # The archive completes FLOWVPM's J with the self-induced term (FLOWVPM skips r=0), so it is a
    # sample of the same continuous field the model decodes; it must match the exact gradient.
    assert rel_J < 0.01, f"particle Jacobian vs autograd Biot-Savart relL2 = {rel_J:.4f}"
    assert rel_J < rel_JT, "archived Jacobian looks transposed (expected J[i, j] = du_i/dx_j)"


# --------------------------------------------------------------------------- rings
def _synthetic_rings(gen, specs, n=400, core=0.05):
    xs, gs, labels = [], [], []
    for k, (z0, R, circ) in enumerate(specs):
        th = torch.rand(n, generator=gen, dtype=torch.float64) * 2 * math.pi
        rr = R + core * torch.randn(n, generator=gen, dtype=torch.float64)
        zz = z0 + core * torch.randn(n, generator=gen, dtype=torch.float64)
        xs.append(torch.stack((rr * torch.cos(th), rr * torch.sin(th), zz), -1))
        tangent = torch.stack((-torch.sin(th), torch.cos(th), torch.zeros_like(th)), -1)
        gs.append(circ * 2 * math.pi * R / n * tangent)
        labels.append(torch.full((n,), k))
    perm = torch.randperm(len(specs) * n, generator=gen)
    return torch.cat(xs)[perm], torch.cat(gs)[perm], torch.cat(labels)[perm]


def test_ring_membership_two_rings():
    gen = torch.Generator().manual_seed(0)
    x, _, true = _synthetic_rings(gen, [(0.0, 1.0, 1.0), (0.75, 0.9, 0.5)])
    label = ring_membership(x, 2)
    assert label.shape == true.shape
    agree = (label == true).float().mean()
    assert max(agree, 1 - agree) == 1.0  # perfect split up to a label permutation
    # lower ring gets label 0 (quantile initialisation is ordered in z)
    assert x[label == 0, 2].mean() < x[label == 1, 2].mean()


def test_ring_diagnostics_position_and_radius():
    gen = torch.Generator().manual_seed(1)
    specs = [(-0.4, 1.2, 1.0), (1.1, 0.8, 0.75)]
    x, gamma, true = _synthetic_rings(gen, specs, n=2000, core=0.02)
    diag = ring_diagnostics(x, gamma, true, 2)
    assert len(diag) == 2
    for (z_k, r_k), (z0, R, _) in zip(diag, specs):
        assert abs(z_k - z0) < 5e-3
        assert abs(r_k - R) < 5e-3
    # an empty ring yields NaNs instead of raising
    z_e, r_e = ring_diagnostics(x, gamma, true, 3)[2]
    assert math.isnan(z_e) and math.isnan(r_e)


@pytest.mark.xfail(strict=True, reason=(
    "Limitation in gino/data/vortex_frames.py: ring_membership clusters on z only and "
    "ring_diagnostics measures radius about the global x=y=0 axis, so a side-by-side pair "
    "(test_outside_regime__free_pair_*: rings centred at x=0 and x=5, same z) is mis-labelled "
    "and the off-axis ring's radius is wrong. Harmless for the coaxial val cases used in training."))
def test_ring_utilities_side_by_side_pair():
    gen = torch.Generator().manual_seed(2)
    x, gamma, true = _synthetic_rings(gen, [(0.0, 1.0, 1.0), (0.0, 1.0, 1.0)], n=400)
    x[true == 1, 0] += 5.0  # second ring translated along x, coplanar with the first
    label = ring_membership(x, 2)
    agree = float((label == true).double().mean())
    assert max(agree, 1 - agree) == 1.0
    for _, r_k in ring_diagnostics(x, gamma, true, 2):
        assert abs(r_k - 1.0) < 0.05
