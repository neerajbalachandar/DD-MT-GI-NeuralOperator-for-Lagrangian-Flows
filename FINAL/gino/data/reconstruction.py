import torch

from .geometry import particle_geometry_features


def rebuild_next_batch(batch, state_phys, particle_field_phys, normalization):
    """Rebuild every model input channel from a predicted physical state and field."""
    out = dict(batch)
    full = batch["particle_features_phys"].clone()
    names = batch["all_feature_names"]
    active = batch["feature_names"]
    idx = {name: names.index(name) for name in names}
    state_names = ("x", "y", "z", "Gamma_x", "Gamma_y", "Gamma_z", "sigma")
    for channel, name in enumerate(state_names):
        full[..., idx[name]] = state_phys[..., channel]
    for channel, name in enumerate(("u_x", "u_y", "u_z")):
        full[..., idx[name]] = particle_field_phys[..., channel]
    grad = particle_field_phys[..., 3:12].reshape(*particle_field_phys.shape[:-1], 3, 3)
    grad_names = [f"gradU_{i}{j}" for i in "xyz" for j in "xyz"]
    for channel, name in enumerate(grad_names):
        full[..., idx[name]] = grad.flatten(-2)[..., channel]

    xyz = state_phys[..., :3].detach().cpu().numpy()
    contexts = batch["pair_context"]
    if isinstance(contexts, list):
        contexts = contexts[0]
    vtk_path = contexts.get("vtk_path", "")
    geom_features = particle_geometry_features(xyz.reshape(-1, 3), vtk_path)
    for name, values in geom_features.items():
        if name in idx:
            tensor = torch.as_tensor(values, dtype=full.dtype, device=full.device).reshape(*state_phys.shape[:-1])
            full[..., idx[name]] = tensor

    phase = float(batch.get("phase_next", 0.0))
    phase_delta = float(batch.get("phase_delta", 0.0))
    if "phase" in idx:
        full[..., idx["phase"]] = phase
    active_indices = [names.index(name) for name in active]
    selected = full[..., active_indices]
    mean = torch.as_tensor(normalization.input_mean, dtype=selected.dtype, device=selected.device)
    std = torch.as_tensor(normalization.input_std, dtype=selected.dtype, device=selected.device).clamp_min(1e-8)
    out["x"] = (selected - mean) / std
    out["particle_features_phys"] = full
    out["state_phys"] = state_phys
    out["input_geom"] = normalization.normalize_positions(state_phys[..., :3]).clamp(0.0, 1.0)
    out["particle_queries"] = out["input_geom"]
    out["global_params"] = batch["global_params"].clone()
    global_names = batch["global_feature_names"]
    if "phase" in global_names:
        out["global_params"][..., global_names.index("phase")] = phase
    if batch.get("rollout_queries") is not None and batch["rollout_queries"].shape[1] > 0:
        out["output_queries"] = batch["rollout_queries"][:, 0]
        out["rollout_queries"] = batch["rollout_queries"][:, 1:]
        out["rollout_field_target"] = batch["rollout_field_target"][:, 1:]
    out["phase_next"] = phase + phase_delta
    return out
