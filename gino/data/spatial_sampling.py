"""Deterministic structured-grid sampling around evolving vorticity cores."""
from __future__ import annotations

import hashlib
import numpy as np


def _structured_layout(coords):
    coords = np.asarray(coords)
    axes = [np.unique(coords[:, i]) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    if np.prod(shape) != len(coords):
        raise ValueError("Vorticity-aware sampling requires a complete structured 3D grid")
    ijk = tuple(np.searchsorted(axes[i], coords[:, i]) for i in range(3))
    flat = np.ravel_multi_index(ijk, shape)
    row = np.empty(len(coords), dtype=np.int64)
    row[flat] = np.arange(len(coords))
    return axes, shape, row.reshape(shape)


def analyze_vorticity_cores(coords, values, config=None):
    """Return retained, physically dilated core mask and interpretable diagnostics."""
    config = config or {}
    coords, values = np.asarray(coords), np.asarray(values)
    omega = values[:, 12:15]
    mag = np.linalg.norm(omega, axis=1)
    valid = np.isfinite(mag) & (mag > 0)
    axes, shape, row_grid = _structured_layout(coords)
    grid = np.zeros(shape, dtype=float)
    grid[tuple(np.searchsorted(axes[i], coords[:, i]) for i in range(3))] = np.where(valid, mag, 0)
    positives = mag[valid]
    if not len(positives):
        return np.zeros(len(coords), dtype=bool), {"threshold": None, "components": 0, "retained_components": 0}
    percentile = float(config.get("vorticity_percentile", 95.0))
    threshold = float(np.percentile(positives, percentile))
    core = (grid >= threshold) & (grid > 0)
    try:
        from scipy import ndimage
        labels, ncomp = ndimage.label(core, structure=np.ones((3, 3, 3), dtype=np.uint8))
        spacing = [float(np.median(np.diff(a))) if len(a) > 1 else 1.0 for a in axes]
        min_nodes = int(config.get("minimum_core_nodes", 8))
        ranked = []
        cell_volume = float(np.prod(spacing))
        for label in range(1, ncomp + 1):
            mask = labels == label
            count = int(mask.sum())
            if count >= min_nodes:
                ranked.append((float((grid[mask] * cell_volume).sum()), count, label))
        ranked.sort(reverse=True)
        retained = ranked[:int(config.get("maximum_core_components", 8))]
        keep_labels = {r[2] for r in retained}
        core = np.isin(labels, list(keep_labels)) if keep_labels else np.zeros(shape, dtype=bool)
        dilation = config.get("core_dilation_physical")
        dilation = 2.0 * max(spacing) if dilation is None else float(dilation)
        if core.any() and dilation > 0:
            distance = ndimage.distance_transform_edt(~core, sampling=spacing)
            dilated = core | (distance <= dilation)
        else:
            dilated = core
        selected = dilated[tuple(np.searchsorted(axes[i], coords[:, i]) for i in range(3))]
        return selected, {"threshold": threshold, "percentile": percentile, "components": int(ncomp),
                         "retained_components": len(retained), "component_integrated_vorticity": [r[0] for r in retained],
                         "core_nodes": int(core.sum()), "dilated_nodes": int(dilated.sum()),
                         "spacing": spacing, "dilation_physical": dilation}
    except ImportError:
        mask = (mag >= threshold) & valid
        return mask, {"threshold": threshold, "percentile": percentile, "components": 1,
                      "retained_components": 1, "core_nodes": int(mask.sum()), "dilated_nodes": int(mask.sum())}


def deterministic_rng(seed, case, frame):
    digest = hashlib.blake2b(f"{int(seed)}|{case}|{frame}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def sample_field_queries(coords, values, budget, config=None, *, case="", frame="", trajectory_bounds=None):
    """Sample unique nodes with 0=uniform, 1=vortex, 2=trajectory labels."""
    cfg = config or {}
    coords = np.asarray(coords)
    n, budget = len(coords), min(int(budget), len(coords))
    mode = str(cfg.get("mode", "regular_grid"))
    if mode in ("regular_grid", "regular"):
        from .preprocessing_pipeline import _field_query_indices
        ids = _field_query_indices(n, budget)
        return ids, np.zeros(len(ids), dtype=np.int8), {"counts": [len(ids), 0, 0], "redistributed": False}
    if mode not in ("vorticity_trajectory_mixture", "uniform"):
        raise ValueError(f"Unknown data.field_sampling.mode: {mode}")
    rng = deterministic_rng(cfg.get("random_seed", 42), case, frame)
    if mode == "uniform":
        axes, _, _ = _structured_layout(coords)
        nbin = max(1, int(round(max(budget, 1) ** (1.0 / 3.0))))
        ijk = np.column_stack([np.searchsorted(axes[i], coords[:, i]) for i in range(3)])
        dims = np.asarray([max(len(a) - 1, 1) for a in axes])
        bins = np.minimum((ijk / dims * nbin).astype(int), nbin - 1)
        strata = np.ravel_multi_index(bins.T, (nbin, nbin, nbin))
        grouped = np.argsort(strata, kind="stable")
        _, first = np.unique(strata[grouped], return_index=True)
        chunks = np.split(grouped, first[1:])
        order = np.concatenate([rng.permutation(chunks[i]) for i in rng.permutation(len(chunks))])
        ids = order[:budget]
        return ids.astype(np.int64), np.zeros(len(ids), dtype=np.int8), {
            "counts": [len(ids), 0, 0], "category_fractions": [1.0, 0.0, 0.0], "redistributed": False}
    core, diagnostics = analyze_vorticity_cores(coords, values, cfg)
    traj = np.zeros(n, dtype=bool)
    if trajectory_bounds is not None and cfg.get("trajectory_envelope_enabled", True):
        lo, hi = np.asarray(trajectory_bounds[0]), np.asarray(trajectory_bounds[1])
        traj = np.all((coords >= lo) & (coords <= hi), axis=1)
    fractions = np.asarray([cfg.get("uniform_fraction", .4), cfg.get("vortex_fraction", .4), cfg.get("trajectory_fraction", .2)], dtype=float)
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("field sampling fractions must be non-negative and sum to one")
    quotas = np.floor(fractions * budget).astype(int)
    for i in np.argsort(-(fractions * budget - quotas))[:budget - quotas.sum()]: quotas[i] += 1
    candidates = [np.arange(n), np.flatnonzero(core), np.flatnonzero(traj)]
    chosen, labels, available = [], [], set()
    # Select by category, excluding earlier picks; shortfalls are filled by the next pools.
    # Shuffle spatial strata, then nodes within each stratum, so the uniform
    # allocation covers the domain even at budgets much smaller than N.
    nbin = max(1, int(round(max(budget * fractions[0], 1) ** (1.0 / 3.0))))
    axes, _, _ = _structured_layout(coords)
    ijk = np.column_stack([np.searchsorted(axes[i], coords[:, i]) for i in range(3)])
    dims = np.asarray([max(len(a) - 1, 1) for a in axes])
    bins = np.minimum((ijk / dims * nbin).astype(int), nbin - 1)
    stratum_ids = np.ravel_multi_index(bins.T, (nbin, nbin, nbin))
    grouped = np.argsort(stratum_ids, kind="stable")
    _, first = np.unique(stratum_ids[grouped], return_index=True)
    chunks = np.split(grouped, first[1:])
    uniform_order = np.concatenate([rng.permutation(chunks[i]) for i in rng.permutation(len(chunks))])
    vortex_pool = candidates[1]
    if len(vortex_pool):
        power = max(float(cfg.get("vorticity_power", 1.0)), 0.0)
        mag = np.linalg.norm(np.asarray(values)[:, 12:15], axis=1)
        prob = np.maximum(mag[vortex_pool], 1e-20) ** power
        prob = prob / prob.sum()
        vortex_order = rng.choice(vortex_pool, size=len(vortex_pool), replace=False, p=prob)
    else:
        vortex_order = np.asarray([], dtype=np.int64)
    orders = [uniform_order, vortex_order, rng.permutation(candidates[2])]
    counts = [0, 0, 0]
    for category in range(3):
        pool = [int(i) for i in orders[category] if int(i) not in available]
        take = min(int(quotas[category]), len(pool))
        chosen.extend(pool[:take]); labels.extend([category] * take); available.update(pool[:take]); counts[category] += take
    redistributed = sum(counts) < budget
    remaining = [[int(i) for i in order if int(i) not in available] for order in orders]
    cursors = [0, 0, 0]
    while len(chosen) < budget:
        progress = False
        for category in (0, 1, 2):
            while cursors[category] < len(remaining[category]) and remaining[category][cursors[category]] in available:
                cursors[category] += 1
            if cursors[category] < len(remaining[category]):
                pool = remaining[category][cursors[category]]
                cursors[category] += 1
                chosen.append(pool); labels.append(category); available.add(pool); counts[category] += 1; progress = True
                if len(chosen) == budget: break
        if not progress:
            break
    result = rng.permutation(len(chosen))
    diagnostics.update(counts=counts, category_fractions=(np.asarray(counts) / max(len(chosen), 1)).tolist(),
                       redistributed=redistributed, trajectory_points=int(traj.sum()))
    return np.asarray(chosen, dtype=np.int64)[result], np.asarray(labels, dtype=np.int8)[result], diagnostics
