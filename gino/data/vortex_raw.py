"""Direct reader for raw FLOWVPM vortex-ring cases (particle + Eulerian HDF5 files).

This replaces the multi-stage merged-frame/npz pipeline for the vortex-ring
datasets with a lossless, per-case export:

* every particle of every saved frame is kept (the legacy archive truncated
  N320/N640 cases to 2048 of 2880/5760 particles, so the stored particles could
  not explain the stored velocity field);
* the exact VPM particle velocity ``u_p`` and velocity gradient ``J_p`` are kept
  as supervision targets (``J[i, j] = du_i/dx_j``; FLOWVPM writes
  ``velocity_gradient_<x_j>[:, i] = du_i/dx_j``);
* frame 0 has no evaluated particle velocity in FLOWVPM output (all zeros), so
  ``particle_velocity_valid[0]`` is ``False`` instead of silently training on zeros;
* Eulerian ``U`` and analytic vorticity ``W`` are kept on the native structured
  grid (x varies fastest, i.e. reshape to ``[nz, ny, nx]``).
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import h5py
import numpy as np

SPLIT_DIRS = {"train": "train", "val": "validate", "test": "test"}


def _steps(case_dir: Path, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.h5$")
    return sorted(int(m.group(1)) for p in case_dir.iterdir() if (m := pattern.match(p.name)))


def discover_cases(raw_root, family: str = "vortexrings") -> list[dict]:
    raw_root = Path(raw_root)
    cases = []
    for split, folder in SPLIT_DIRS.items():
        base = raw_root / folder / family
        if not base.is_dir():
            continue
        for case_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            meta = {}
            toml_path = case_dir / "case.toml"
            if toml_path.is_file():
                with open(toml_path, "rb") as handle:
                    meta = tomllib.load(handle)
            cases.append({"name": case_dir.name, "split": split, "path": str(case_dir), "meta": meta})
    return cases


def read_case(case_dir, max_frames: int | None = None) -> dict:
    """Read one raw case into float32 arrays (frames ordered by solver step)."""
    case_dir = Path(case_dir)
    p_steps = _steps(case_dir, "pfield")
    f_steps = _steps(case_dir, "fields_fdom")
    steps = [s for s in p_steps if s in set(f_steps)]
    if len(steps) < 2:
        raise ValueError(f"{case_dir} has fewer than two matched particle/field frames")
    if max_frames:
        steps = steps[:max_frames]
    times, X, G, S, Up, Jp = [], [], [], [], [], []
    for step in steps:
        with h5py.File(case_dir / f"pfield.{step}.h5", "r") as h:
            times.append(float(h["t"][()]))
            X.append(h["X"][()])
            G.append(h["Gamma"][()])
            S.append(h["sigma"][()])
            Up.append(h["velocity"][()])
            # J[n, i, j] = du_i / dx_j
            Jp.append(np.stack([h["velocity_gradient_x"][()], h["velocity_gradient_y"][()],
                                h["velocity_gradient_z"][()]], axis=-1))
    counts = {len(x) for x in X}
    if len(counts) != 1:
        raise ValueError(f"{case_dir} has a varying particle count {sorted(counts)}; "
                         "fixed row identity is required for rollout targets")
    nodes, Ug, Wg = None, [], []
    for step in steps:
        with h5py.File(case_dir / f"fields_fdom.{step}.h5", "r") as h:
            n = h["nodes"][()]
            n = n.T if n.shape[0] == 3 and n.shape[1] != 3 else n
            if nodes is None:
                nodes = n
            elif not np.allclose(nodes, n):
                raise ValueError(f"{case_dir}: Eulerian mesh moves between frames; not supported")
            Ug.append(h["U"][()])
            Wg.append(h["W"][()])
    axes = [np.unique(nodes[:, k]) for k in range(3)]
    shape_zyx = (len(axes[2]), len(axes[1]), len(axes[0]))
    if int(np.prod(shape_zyx)) != len(nodes):
        raise ValueError(f"{case_dir}: field nodes are not a complete structured grid")
    grid = nodes.reshape(shape_zyx + (3,))
    if not (np.allclose(grid[0, 0, :, 0], axes[0]) and np.allclose(grid[0, :, 0, 1], axes[1])
            and np.allclose(grid[:, 0, 0, 2], axes[2])):
        raise ValueError(f"{case_dir}: unexpected node ordering (expected x fastest, then y, then z)")
    Up = np.asarray(Up, dtype=np.float32)
    valid = np.abs(Up).reshape(len(steps), -1).max(axis=1) > 0
    return {
        "steps": np.asarray(steps, dtype=np.int64),
        "time": np.asarray(times, dtype=np.float64),
        "x": np.asarray(X, dtype=np.float32),
        "gamma": np.asarray(G, dtype=np.float32),
        "sigma": np.asarray(S, dtype=np.float32),
        "particle_velocity": Up,
        "particle_jacobian": np.asarray(Jp, dtype=np.float32),
        "particle_velocity_valid": valid,
        "grid_nodes": nodes.astype(np.float32),
        "grid_shape_zyx": np.asarray(shape_zyx, dtype=np.int64),
        "grid_velocity": np.asarray(Ug, dtype=np.float32),
        "grid_vorticity": np.asarray(Wg, dtype=np.float32),
    }


def winckelmans_velocity(q, x, gamma, sigma, chunk: int = 4096):
    """Regularised (Winckelmans) Biot-Savart velocity of particles (x, gamma, sigma) at q (torch)."""
    import torch
    out = []
    s2 = (sigma * sigma).unsqueeze(0)
    for start in range(0, len(q), chunk):
        r = q[start:start + chunk, None, :] - x[None]
        r2 = (r * r).sum(-1)
        f = (r2 + 2.5 * s2) / (r2 + s2).pow(2.5)
        out.append(-(f.unsqueeze(-1) * torch.cross(r, gamma[None].expand_as(r), dim=-1)).sum(1) / (4.0 * np.pi))
    return torch.cat(out)


def winckelmans_velocity_jacobian(q, x, gamma, sigma, chunk: int = 1024):
    """Exact regularised Biot-Savart u [Q,3] and J [Q,3,3] (J[i,j] = du_i/dq_j), torch.

    u = -1/(4 pi) sum_p f(r^2) (r x Gamma_p),  f = (r^2 + 2.5 s^2) / (r^2 + s^2)^(5/2), r = q - x_p
    J_ij = -1/(4 pi) sum_p [ f eps_ijl Gamma_l + 2 f'(r^2) (r x Gamma)_i r_j ]
    The self pair (q = x_p) contributes the continuous-field self term.  Used to make exact
    targets for perturbed (off-trajectory) particle sets during training augmentation.
    """
    import torch
    us, Js = [], []
    s2 = (sigma * sigma).unsqueeze(0)
    levi = torch.zeros(3, 3, 3, dtype=q.dtype, device=q.device)
    levi[0, 1, 2] = levi[1, 2, 0] = levi[2, 0, 1] = 1.0
    levi[0, 2, 1] = levi[2, 1, 0] = levi[1, 0, 2] = -1.0
    G_skew = torch.einsum("ijl,pl->pij", levi, gamma)                    # eps_ijl Gamma_l per particle
    for start in range(0, len(q), chunk):
        r = q[start:start + chunk, None, :] - x[None]
        r2 = (r * r).sum(-1)
        d = r2 + s2
        f = (r2 + 2.5 * s2) / d.pow(2.5)
        fp = d.pow(-2.5) - 2.5 * (r2 + 2.5 * s2) / d.pow(3.5)
        rxg = torch.cross(r, gamma[None].expand_as(r), dim=-1)
        us.append(-(f.unsqueeze(-1) * rxg).sum(1) / (4.0 * np.pi))
        J = torch.einsum("qp,pij->qij", f, G_skew) + 2.0 * torch.einsum("qp,qpi,qpj->qij", fp, rxg, r)
        Js.append(-J / (4.0 * np.pi))
    return torch.cat(us), torch.cat(Js)


def self_induced_jacobian(gamma, sigma):
    """Winckelmans self term 2.5/(4 pi sigma^3) [Gamma]_x that FLOWVPM omits from particle J (numpy)."""
    g = np.asarray(gamma, dtype=np.float64)
    sk = np.zeros(g.shape[:-1] + (3, 3))
    sk[..., 0, 1], sk[..., 0, 2] = -g[..., 2], g[..., 1]
    sk[..., 1, 0], sk[..., 1, 2] = g[..., 2], -g[..., 0]
    sk[..., 2, 0], sk[..., 2, 1] = -g[..., 1], g[..., 0]
    return (2.5 / (4.0 * np.pi * np.asarray(sigma, dtype=np.float64) ** 3))[..., None, None] * sk


def add_exact_grid_velocity(path, device="cuda", verbose=True):
    """Replace FLOWVPM's grid U with the exact Biot-Savart field of the stored particles.

    FLOWVPM's computefluiddomain evaluates U after injecting ~32k grid-node probe
    particles (small Gamma, sigma = 0.5 mean sigma) into the particle field, so the
    stored far-field velocity is 15-28% off the field of the actual vortex particles.
    The original array is kept as ``grid_velocity_flowvpm``.  Particle J is completed
    with the self-induced term so that it is a sample of the same continuous field.
    """
    import torch
    path = Path(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if "grid_velocity_flowvpm" in arrays:
        return
    nodes = torch.as_tensor(arrays["grid_nodes"], device=device, dtype=torch.float32)
    exact = np.empty_like(arrays["grid_velocity"])
    for t in range(len(arrays["x"])):
        x = torch.as_tensor(arrays["x"][t], device=device)
        g = torch.as_tensor(arrays["gamma"][t], device=device)
        s = torch.as_tensor(arrays["sigma"][t], device=device)
        exact[t] = winckelmans_velocity(nodes, x, g, s).cpu().numpy()
    orig = arrays["grid_velocity"]
    rel = float(np.linalg.norm(exact - orig) / np.linalg.norm(exact))
    arrays["grid_velocity_flowvpm"] = orig
    arrays["grid_velocity"] = exact.astype(np.float32)
    arrays["particle_jacobian_flowvpm"] = arrays["particle_jacobian"]
    full = arrays["particle_jacobian"].astype(np.float64) + self_induced_jacobian(arrays["gamma"], arrays["sigma"])
    full[~arrays["particle_velocity_valid"]] = 0.0
    arrays["particle_jacobian"] = full.astype(np.float32)
    arrays["exact_grid_velocity_vs_flowvpm_rel_l2"] = np.asarray(rel)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    if verbose:
        print(f"[exact-U] {path.stem}: FLOWVPM grid U vs exact Biot-Savart relL2 = {rel:.4f}", flush=True)


def export_cases(raw_root, out_dir, family: str = "vortexrings", overwrite: bool = False,
                 include=None, verbose: bool = True, exact_fields: bool = True, device: str = "cuda") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"family": family, "raw_root": str(Path(raw_root).resolve()), "cases": []}
    for case in discover_cases(raw_root, family):
        if include and case["name"] not in include:
            continue
        path = out_dir / f"{case['name']}.npz"
        if overwrite or not path.is_file():
            arrays = read_case(case["path"])
            np.savez(path, **arrays)
            if verbose:
                print(f"[export] {case['split']:5s} {case['name']}: frames={len(arrays['steps'])} "
                      f"particles={arrays['x'].shape[1]} grid={tuple(arrays['grid_shape_zyx'])}", flush=True)
        if exact_fields:
            add_exact_grid_velocity(path, device=device, verbose=verbose)
        with np.load(path) as z:
            n_frames, n_particles = z["x"].shape[:2]
            dt = np.diff(z["time"])
            grid = z["grid_shape_zyx"].tolist()
        meta = case["meta"]
        manifest["cases"].append({
            "name": case["name"], "split": case["split"], "file": path.name,
            "frames": int(n_frames), "particles": int(n_particles), "grid_shape_zyx": grid,
            "dt_mean": float(dt.mean()), "dt_min": float(dt.min()), "dt_max": float(dt.max()),
            "evaluation_categories": list(meta.get("evaluation_categories", [])),
            "save_every_steps": meta.get("save_every_steps"),
            "spacing_over_radius": meta.get("spacing_over_radius"),
            "circulation_ratio": meta.get("circulation_ratio"),
        })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir / "manifest.json"
