import torch
from .physics import physical_transition


def predict_next_state(model, batch, latent_grid, normalization, dt=None):
    """Canonical model + field denormalization + physical and residual transition."""
    field_queries = batch["output_queries"]
    particle_queries = batch.get("particle_queries", batch["input_geom"])
    residual_norm, particle_field_norm = model(batch["input_geom"], latent_grid, particle_queries,
                                               batch["x"], batch["global_params"], batch_dict=batch)
    if particle_queries.shape[1] == field_queries.shape[1] and torch.equal(particle_queries, field_queries):
        field_norm = particle_field_norm
    else:
        _, field_norm = model(batch["input_geom"], latent_grid, field_queries,
                              batch["x"], batch["global_params"], batch_dict=batch)
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
