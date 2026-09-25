import torch
from gino.data.normalization import NormalizationStats
from gino.dynamics.rollout import autoregressive_rollout, training_rollout
from gino.dynamics.state_transition import predict_next_state


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


def test_state_transition_exposes_physical_base_and_residual():
    class ConstantModel:
        def __call__(self, geom, latent, query, x, global_params, batch_dict=None):
            residual = torch.ones((*x.shape[:2], 7), dtype=x.dtype, device=x.device)
            field = torch.zeros((*query.shape[:2], 12), dtype=x.dtype, device=x.device)
            field[..., :3] = torch.tensor([1.0, 2.0, 3.0], device=x.device)
            field[..., 3:12] = torch.eye(3, device=x.device).reshape(9)
            return residual, field

    stats = NormalizationStats(torch.zeros(1), torch.ones(1), torch.zeros(7), torch.ones(7),
                               torch.zeros(12), torch.ones(12), torch.zeros(3), torch.ones(3))
    state = torch.tensor([[[0., 0., 0., 1., 2., 3., 0.5]]])
    batch = {"input_geom": torch.zeros(1, 1, 3), "particle_queries": torch.zeros(1, 1, 3),
             "output_queries": torch.ones(1, 2, 3), "x": torch.zeros(1, 1, 1),
             "global_params": torch.zeros(1, 1), "state_phys": state, "dt": 0.1}
    result = predict_next_state(ConstantModel(), batch, torch.zeros(1, 1, 3), stats)
    torch.testing.assert_close(result["base_state"][..., :3], torch.tensor([[[0.1, 0.2, 0.3]]]))
    torch.testing.assert_close(result["predicted_state"][..., 3:6], torch.tensor([[[2.1, 3.2, 4.3]]]))
    torch.testing.assert_close(result["predicted_state"][..., 6], torch.tensor([[1.5]]))
    assert result["field_norm"].shape == (1, 2, 12)
    assert result["field_phys"].shape == (1, 2, 12)


def test_training_rollout_keeps_gradients_across_steps():
    class WeightedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, geom, latent, query, x, global_params, batch_dict=None):
            residual = self.weight.expand(*x.shape[:2], 7)
            field = torch.zeros((*query.shape[:2], 12), dtype=x.dtype, device=x.device)
            return residual, field

    stats = NormalizationStats(torch.zeros(1), torch.ones(1), torch.zeros(7), torch.ones(7),
                               torch.zeros(12), torch.ones(12), torch.zeros(3), torch.ones(3))
    batch = {"input_geom": torch.zeros(1, 1, 3), "particle_queries": torch.zeros(1, 1, 3),
             "output_queries": torch.zeros(1, 1, 3), "x": torch.zeros(1, 1, 1),
             "global_params": torch.zeros(1, 1), "state_phys": torch.zeros(1, 1, 7), "dt": 0.0}

    def rebuild(old, state, field):
        nxt = dict(old)
        nxt["state_phys"] = state
        return nxt

    model = WeightedModel()
    states, _ = training_rollout(model, batch, torch.zeros(1, 1, 3), stats, 2, rebuild)
    states.sum().backward()
    assert model.weight.grad is not None
    assert model.weight.grad.abs().item() > 0
