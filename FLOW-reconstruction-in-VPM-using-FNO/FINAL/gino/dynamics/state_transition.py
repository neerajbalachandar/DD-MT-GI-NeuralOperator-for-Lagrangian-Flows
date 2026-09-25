import torch


def predict_next_state(model, batch, latent_grid, normalization, dt=None):
    """Canonical model + physical base + learned residual transition."""
    residual_norm, field_norm = model(batch["input_geom"], latent_grid, batch["output_queries"], batch["x"], batch["global_params"], batch_dict=batch)
    residual = normalization.denormalize_residual(residual_norm)[..., :7]
    field = normalization.denormalize_field(field_norm)
    velocity, grad = field[..., :3], field[..., 3:12].reshape(*field.shape[:-1], 3, 3)
    state = batch["state_phys"]
    step = batch.get("dt") if dt is None else dt
    while torch.is_tensor(step) and step.ndim < state.ndim:
        step = step.unsqueeze(-1)
    base = state.clone()
    base[..., :3] = state[..., :3] + step * velocity
    base[..., 3:6] = state[..., 3:6] + step * torch.einsum("...ij,...j->...i", grad, state[..., 3:6])
    return base + residual, base, residual, field_norm, field
