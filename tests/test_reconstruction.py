import numpy as np
import torch

from gino.data.normalization import NormalizationStats
from gino.data.reconstruction import rebuild_next_batch
from gino.dynamics.rollout import inference_rollout


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
             "pair_context": {"vtk_path": "mock.vtk", "vtk_path_tp1": "next.vtk",
                              "phase_t": 0.3, "phase_tp1": 0.4, "phase_delta": 0.1}, "phase_next": 0.4, "phase_delta": 0.1,
             "rollout_queries": torch.zeros(1, 1, 2, 3), "rollout_field_targets": torch.zeros(1, 2, 2, 12),
             "rollout_contexts": [], "rollout_pair_ids": [], "rollout_phases": torch.tensor([[0.4]])}
    stats = NormalizationStats(np.zeros(6), np.ones(6), np.zeros(7), np.ones(7),
                               np.zeros(12), np.ones(12), np.zeros(3), np.ones(3))
    state = torch.tensor([[[0.2, 0.3, 0.4, 1., 2., 3., 0.7],
                           [0.5, 0.6, 0.7, 4., 5., 6., 0.8]]])
    field = torch.zeros(1, 2, 12)
    field[..., 0] = 2.0
    field[..., 3] = 3.0
    rebuilt = rebuild_next_batch(batch, state, field, stats)
    assert rebuilt["geometry_differentiable"] is False
    assert "VTK/NumPy" in rebuilt["geometry_gradient_boundary"]
    torch.testing.assert_close(rebuilt["x"][..., 0], torch.full((1, 2), 2.0))
    torch.testing.assert_close(rebuilt["x"][..., 1], state[..., 6])
    torch.testing.assert_close(rebuilt["x"][..., 2], torch.full((1, 2), 0.25))
    torch.testing.assert_close(rebuilt["x"][..., 3], state[..., 3])
    torch.testing.assert_close(rebuilt["x"][..., 4], torch.full((1, 2), 3.0))
    torch.testing.assert_close(rebuilt["x"][..., 5], torch.full((1, 2), 0.4))
    torch.testing.assert_close(rebuilt["global_params"], torch.tensor([[0.4]]))
    torch.testing.assert_close(rebuilt["output_queries"], torch.zeros(1, 2, 3))


def test_h2_rollout_uses_generated_state_and_advancing_context(monkeypatch):
    monkeypatch.setattr("gino.data.reconstruction.particle_geometry_features",
                        lambda xyz, path: {"geom_dist": np.full(len(xyz), 0.1, dtype=np.float32)})
    names = ["x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma", "u_x", "u_y", "u_z",
             *[f"gradU_{i}{j}" for i in "xyz" for j in "xyz"], "geom_dist", "phase"]
    active = ["u_x", "sigma", "geom_dist", "Gamma_x", "phase"]
    context0 = {"case": "c", "frame_t": "0", "frame_tp1": "1", "phase_t": 0., "phase_tp1": .5,
                "phase_delta": .5, "vtk_path": "v0", "vtk_path_tp1": "v1", "dt": .1}
    context1 = {"case": "c", "frame_t": "1", "frame_tp1": "2", "phase_t": .5, "phase_tp1": 1.,
                "phase_delta": .5, "vtk_path": "v1", "vtk_path_tp1": "v2", "dt": .1}
    full = torch.zeros(1, 2, len(names))
    state = torch.zeros(1, 2, 7)
    batch = {"input_geom": torch.zeros(1, 2, 3), "particle_queries": torch.zeros(1, 2, 3),
             "output_queries": torch.zeros(1, 2, 3), "x": torch.zeros(1, 2, len(active)),
             "global_params": torch.zeros(1, 1), "state_phys": state, "dt": .1,
             "particle_features_phys": full, "all_feature_names": names, "feature_names": active,
             "global_feature_names": ["phase"], "input_mean": np.zeros(len(active)),
             "input_std": np.ones(len(active)), "pair_context": context0,
             "rollout_contexts": [context1], "rollout_pair_ids": [1],
             "rollout_queries": torch.zeros(1, 1, 2, 3),
             "rollout_field_targets": torch.zeros(1, 2, 2, 12),
             "rollout_phases": torch.tensor([[.5, 1.]]),
             "particle_correspondence": "canonical row ordering"}
    stats = NormalizationStats(np.zeros(len(active)), np.ones(len(active)), np.zeros(7), np.ones(7),
                               np.zeros(12), np.ones(12), np.zeros(3), np.ones(3))

    class ConstantStep(torch.nn.Module):
        def encode_process(self, geom, latent, x, global_params):
            return {"shape": x.shape[:2]}

        def decode_particle(self, encoded):
            return torch.zeros((*encoded["shape"], 7))

        def decode_field(self, encoded, query):
            field = torch.zeros((query.shape[0], query.shape[1], 12))
            field[..., 0] = 1.0
            return field

    target_mutated = dict(batch)
    target_mutated["rollout_state_targets"] = torch.full((1, 2, 2, 7), 1e6)
    latent = torch.zeros(1, 1, 3)
    rebuild = lambda old, predicted, field: rebuild_next_batch(old, predicted, field, stats)
    states, _ = inference_rollout(ConstantStep(), target_mutated, latent, stats, 2, rebuild)
    torch.testing.assert_close(states[0, :, :, 0], torch.tensor([[.1, .1], [.2, .2]]))
    rebuilt = rebuild_next_batch(batch, states[:, 0], torch.cat((torch.ones(1, 2, 1), torch.zeros(1, 2, 11)), -1), stats)
    assert rebuilt["pair_context"]["frame_t"] == "1"
    assert rebuilt["pair_context"]["vtk_path"] == "v1"
    assert rebuilt["global_params"][0, 0].item() == .5
