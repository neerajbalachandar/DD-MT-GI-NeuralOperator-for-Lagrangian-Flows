import torch
from gino.data.normalization import NormalizationStats
from gino.dynamics.rollout import autoregressive_rollout


class IncrementModel:
    def __call__(self, geom, latent, query, x, global_params, batch_dict=None):
        return torch.ones((*x.shape[:2], 7)), torch.zeros((*query.shape[:2], 12))


def test_rollout_propagates_predictions_and_refreshes_geometry():
    stats = NormalizationStats(torch.zeros(1), torch.ones(1), torch.zeros(7), torch.ones(7),
                               torch.zeros(12), torch.ones(12), torch.zeros(3), torch.ones(3))
    batch = {"input_geom": torch.zeros(1, 2, 3), "output_queries": torch.zeros(1, 2, 3),
             "x": torch.zeros(1, 2, 1), "global_params": torch.zeros(1, 1),
             "state_phys": torch.zeros(1, 2, 7), "dt": torch.tensor([0.])}
    seen = []
    def rebuild(old, predicted, field):
        new = dict(old)
        new["state_phys"] = predicted
        new["input_geom"] = predicted[..., :3].clone()
        seen.append(new["input_geom"].clone())
        return new
    states, _ = autoregressive_rollout(IncrementModel(), batch, torch.zeros(1, 1, 3), stats, 2, rebuild)
    assert states.shape == (1, 2, 2, 7)
    assert torch.all(states[:, 1] > states[:, 0])
    assert len(seen) == 2
    assert not torch.equal(seen[0], batch["input_geom"])
