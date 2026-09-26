import torch
from .physics import physical_transition


def predict_next_state(model, batch, latent_grid, normalization, dt=None):
    """Canonical model + field denormalization + physical and residual transition."""
    encoded = model.encode_process(batch["input_geom"], latent_grid, batch["x"], batch["global_params"])
    residual_norm = model.decode_particle(encoded)
    particle_field_norm = model.decode_field(encoded, batch.get("particle_queries", batch["input_geom"]))
    field_norm = model.decode_field(encoded, batch["output_queries"])
    residual = normalization.denormalize_residual(residual_norm)[..., :7]
    field = normalization.denormalize_field(particle_field_norm)
    velocity, grad = field[..., :3], field[..., 3:12].reshape(*field.shape[:-1], 3, 3)
    state = batch["state_phys"]
    step = batch.get("dt") if dt is None else dt
    while torch.is_tensor(step) and step.ndim < state.ndim:
        step = step.unsqueeze(-1)
    base = physical_transition(state, velocity, grad, step)
    return {"predicted_state": base + residual, "base_state": base, "residual": residual,
            "field_norm": field_norm, "field_phys": normalization.denormalize_field(field_norm),
            "particle_field_norm": particle_field_norm, "particle_field_phys": field}
