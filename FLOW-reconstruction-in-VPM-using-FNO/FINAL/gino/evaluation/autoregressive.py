from gino.dynamics.rollout import autoregressive_rollout
from .metrics import state_component_metrics


def evaluate_autoregressive(model, initial_batch, latent_grid, normalization, targets, horizons, rebuild_batch):
    predicted, _ = autoregressive_rollout(model, initial_batch, latent_grid, normalization, max(map(int, horizons)), rebuild_batch)
    results = []
    for horizon in horizons:
        h = int(horizon)
        if h <= targets.shape[1]:
            results.append({"horizon": h, **state_component_metrics(predicted[:, h - 1].cpu().numpy(), targets[:, h - 1].cpu().numpy())})
    return results
