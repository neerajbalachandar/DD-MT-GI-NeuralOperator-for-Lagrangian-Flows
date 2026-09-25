import numpy as np
import torch

from gino.data.normalization import NormalizationStats
from gino.data.reconstruction import rebuild_next_batch


def test_rebuild_updates_physical_features_geometry_and_phase(monkeypatch):
    monkeypatch.setattr("gino.data.reconstruction.particle_geometry_features",
                        lambda xyz, path: {"geom_dist": np.full(len(xyz), 0.25, dtype=np.float32)})
    names = ["x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma", "u_x", "u_y", "u_z",
             "gradU_xx", "gradU_xy", "gradU_xz", "gradU_yx", "gradU_yy", "gradU_yz",
             "gradU_zx", "gradU_zy", "gradU_zz", "geom_dist", "phase"]
    active = ["u_x", "sigma", "geom_dist", "Gamma_x", "gradU_xx", "phase"]
    full = torch.zeros(1, 2, len(names))
    batch = {"particle_features_phys": full, "all_feature_names": names, "feature_names": active,
             "global_feature_names": ["phase"], "global_params": torch.zeros(1, 1),
             "pair_context": {"vtk_path": "mock.vtk"}, "phase_next": 0.4, "phase_delta": 0.1,
             "rollout_queries": torch.zeros(1, 1, 2, 3), "rollout_field_target": torch.zeros(1, 1, 2, 12)}
    stats = NormalizationStats(np.zeros(6), np.ones(6), np.zeros(7), np.ones(7),
                               np.zeros(12), np.ones(12), np.zeros(3), np.ones(3))
    state = torch.tensor([[[0.2, 0.3, 0.4, 1., 2., 3., 0.7],
                           [0.5, 0.6, 0.7, 4., 5., 6., 0.8]]])
    field = torch.zeros(1, 2, 12)
    field[..., 0] = 2.0
    field[..., 3] = 3.0
    rebuilt = rebuild_next_batch(batch, state, field, stats)
    torch.testing.assert_close(rebuilt["x"][..., 0], torch.full((1, 2), 2.0))
    torch.testing.assert_close(rebuilt["x"][..., 1], state[..., 6])
    torch.testing.assert_close(rebuilt["x"][..., 2], torch.full((1, 2), 0.25))
    torch.testing.assert_close(rebuilt["x"][..., 3], state[..., 3])
    torch.testing.assert_close(rebuilt["x"][..., 4], torch.full((1, 2), 3.0))
    torch.testing.assert_close(rebuilt["x"][..., 5], torch.full((1, 2), 0.4))
    torch.testing.assert_close(rebuilt["global_params"], torch.tensor([[0.4]]))
    torch.testing.assert_close(rebuilt["output_queries"], torch.zeros(1, 2, 3))
