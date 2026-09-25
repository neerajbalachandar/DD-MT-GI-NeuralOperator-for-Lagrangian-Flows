import torch
from .state_transition import predict_next_state


@torch.no_grad()
def autoregressive_rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch):
    """Closed loop. rebuild_batch must rebuild features and geometry from predicted states."""
    batch = initial_batch
    states, fields = [], []
    for _ in range(int(steps)):
        predicted, _, _, _, field = predict_next_state(model, batch, latent_grid, normalization)
        states.append(predicted)
        fields.append(field)
        batch = rebuild_batch(batch, predicted, field)
    return torch.stack(states, dim=1), torch.stack(fields, dim=1)
