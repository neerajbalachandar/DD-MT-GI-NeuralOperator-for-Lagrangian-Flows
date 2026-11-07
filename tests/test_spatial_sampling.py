import numpy as np

from gino.data.spatial_sampling import sample_field_queries
from gino.evaluation.vortex_geometry import detect_ring_geometry, smooth_axis
from gino.training.losses import grouped_field_loss


def _ring_grid(z_centers=(0.0,), n=18):
    axis = np.linspace(-1.0, 1.0, n)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    x, y, z = xyz.T
    radial = np.sqrt(x*x + y*y)
    omega = np.zeros_like(xyz)
    magnitude = np.zeros(len(xyz))
    for center in z_centers:
        magnitude += np.exp(-((radial - .5)**2 + (z-center)**2) / (2*.10**2))
    omega[:, 1] = magnitude
    values = np.zeros((len(xyz), 15), dtype=np.float32)
    values[:, 12:15] = omega
    return xyz, values


def test_adaptive_sampler_is_unique_deterministic_and_category_aware():
    xyz, values = _ring_grid()
    cfg = {"mode": "vorticity_trajectory_mixture", "uniform_fraction": .4,
           "vortex_fraction": .4, "trajectory_fraction": .2,
           "vorticity_percentile": 80, "minimum_core_nodes": 2,
           "maximum_core_components": 8, "random_seed": 7}
    envelope = (np.array([-.7, -.7, -.5]), np.array([.7, .7, .5]))
    ids, cats, meta = sample_field_queries(xyz, values, 400, cfg, case="ring", frame=1,
                                           trajectory_bounds=envelope)
    ids2, cats2, _ = sample_field_queries(xyz, values, 400, cfg, case="ring", frame=1,
                                          trajectory_bounds=envelope)
    assert len(ids) == 400 and len(np.unique(ids)) == 400
    assert np.array_equal(ids, ids2) and np.array_equal(cats, cats2)
    assert np.bincount(cats, minlength=3).tolist() == [160, 160, 80]
    assert np.all((xyz[ids[cats == 2]] >= envelope[0]) & (xyz[ids[cats == 2]] <= envelope[1]))
    assert np.all(np.ptp(xyz[ids[cats == 0]], axis=0) > 1.0)
    assert meta["retained_components"] >= 1
    ids_other, _, _ = sample_field_queries(xyz, values, 400, cfg, case="ring", frame=2,
                                           trajectory_bounds=envelope)
    assert not np.array_equal(ids, ids_other)


def test_weak_vorticity_redistributes_without_duplicates():
    xyz, values = _ring_grid()
    values[:, 12:15] = 0
    cfg = {"mode": "vorticity_trajectory_mixture", "uniform_fraction": .4,
           "vortex_fraction": .4, "trajectory_fraction": .2, "random_seed": 4}
    ids, categories, diagnostics = sample_field_queries(xyz, values, 300, cfg, case="weak", frame=0)
    assert len(ids) == len(np.unique(ids)) == 300
    assert diagnostics["components"] == 0
    assert diagnostics["redistributed"]
    assert categories.sum() == 0


def test_ring_geometry_finds_axis_and_separated_axial_peaks():
    xyz, values = _ring_grid((-0.45, 0.45), n=30)
    result = detect_ring_geometry(xyz, values[:, 12:15], {
        "vorticity_percentile": 90, "minimum_core_nodes": 2,
        "maximum_core_components": 8, "minimum_peak_separation_physical": .4,
        "peak_prominence_fraction": .15,
    })
    assert result["valid"]
    center = np.asarray(result["axis_center_instantaneous"])
    assert np.linalg.norm(center[:2]) < .15
    assert len(result["ring_axial_locations"]) == 2
    assert np.allclose(smooth_axis(None, center, .2), center)


def test_category_losses_renormalize_for_empty_categories():
    import torch
    target = torch.zeros((1, 4, 15))
    pred = torch.ones_like(target)
    categories = torch.tensor([[0, 0, 1, 1]])
    total, groups = grouped_field_loss(pred, target, categories=categories,
                                      category_weights=[.4, .4, .2])
    assert torch.isfinite(total)
    assert torch.allclose(groups["velocity"], torch.tensor(1.0))
    assert "velocity_category_2" not in groups

