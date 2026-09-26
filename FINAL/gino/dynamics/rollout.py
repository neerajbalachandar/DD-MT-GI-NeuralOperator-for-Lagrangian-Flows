import torch
from .state_transition import predict_next_state


def _rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch,
             detach_generated=False, store_on_cpu=False, collect_fields=True):
    steps = int(steps)
    available = 1 + len(initial_batch.get("rollout_contexts", ()))
    if steps < 1:
        raise ValueError("Rollout requires at least one prediction step")
    if steps > available:
        raise ValueError(f"Requested {steps} rollout steps, but only {available} target/context steps exist")
    batch = initial_batch
    states, fields = [], []
    for step in range(steps):
        result = predict_next_state(model, batch, latent_grid, normalization)
        predicted, field = result["predicted_state"], result["particle_field_phys"]
        query_field = result["field_phys"]
        states.append(predicted.detach().cpu() if store_on_cpu else predicted)
        if collect_fields:
            fields.append(query_field.detach().cpu() if store_on_cpu else query_field)
        if step + 1 < steps:
            batch = rebuild_batch(batch, predicted.detach() if detach_generated else predicted,
                                  field.detach() if detach_generated else field)
    return torch.stack(states, dim=1), torch.stack(fields, dim=1) if fields else None


@torch.inference_mode()
def inference_rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch, store_on_cpu=False):
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        return _rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch,
                        store_on_cpu=store_on_cpu)
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(was_training)


def training_rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch):
    """Differentiable BPTT loop; predicted inputs remain attached to the graph."""
    return _rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch,
                    collect_fields=False)


def pushforward_rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch):
    """Train on generated inputs while detaching each transition boundary."""
    return _rollout(model, initial_batch, latent_grid, normalization, steps, rebuild_batch,
                    detach_generated=True, collect_fields=False)


autoregressive_rollout = inference_rollout
