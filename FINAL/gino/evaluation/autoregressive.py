from gino.dynamics.rollout import inference_rollout
from .metrics import state_component_metrics


def evaluate_autoregressive(model, initial_batch, latent_grid, normalization, targets, horizons, rebuild_batch):
    predicted, _ = inference_rollout(model, initial_batch, latent_grid, normalization,
                                     max(map(int, horizons)), rebuild_batch, store_on_cpu=True)
    results = []
    for horizon in horizons:
        h = int(horizon)
        if h <= targets.shape[1]:
            truth = targets[:, h - 1]
            truth = truth.cpu().numpy() if hasattr(truth, "cpu") else truth
            results.append({"horizon": h, **state_component_metrics(predicted[:, h - 1].numpy(), truth)})
    return results
