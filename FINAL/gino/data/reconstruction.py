import torch

from .geometry import particle_geometry_features


def rebuild_next_batch(batch, state_phys, particle_field_phys, normalization, differentiable=False):
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

    # VTK/NumPy geometry is an external non-differentiable boundary. Other
    # reconstructed features and normalized particle coordinates retain their graph.
    xyz = state_phys[..., :3].detach().cpu().numpy()
    contexts = batch.get("rollout_contexts", [])
    current_context = batch["pair_context"]
    context = contexts[0] if contexts else dict(current_context)
    if isinstance(context, list):
        context = context[0]
    vtk_path = context.get("vtk_path", "")
    geom_features = particle_geometry_features(xyz.reshape(-1, 3), vtk_path)
    for name, values in geom_features.items():
        if name in idx:
            tensor = torch.as_tensor(values, dtype=full.dtype, device=full.device).reshape(*state_phys.shape[:-1])
            full[..., idx[name]] = tensor

    phase = float(context.get("phase_t", batch.get("phase_next", 0.0)))
    phase_delta = float(context.get("phase_delta", batch.get("phase_delta", 0.0)))
    if not contexts:
        phase = float(batch.get("phase_next", phase))
        context.update({"frame_t": current_context.get("frame_tp1", current_context.get("frame_t", "")),
                        "phase_t": phase, "phase_tp1": phase + phase_delta,
                        "vtk_path": current_context.get("vtk_path_tp1", "")})
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
    out["pair_context"] = context
    out["dt"] = float(context.get("dt", batch.get("dt", 0.0034)))
    out["geometry_differentiable"] = False
    out["reconstruction_mode"] = "training_bptt" if differentiable else "inference_or_detached_pushforward"
    out["geometry_gradient_boundary"] = "stopped at VTK/NumPy nearest-surface lookup; state, field, and normalized-coordinate paths retain gradients in BPTT"
    out["particle_correspondence"] = batch.get("particle_correspondence", "canonical row ordering")
    if batch.get("rollout_queries") is not None and batch["rollout_queries"].shape[1] > 0:
        out["output_queries"] = batch["rollout_queries"][:, 0]
        out["rollout_queries"] = batch["rollout_queries"][:, 1:]
        out["rollout_field_targets"] = batch["rollout_field_targets"][:, 1:]
        out["rollout_contexts"] = contexts[1:]
        if batch.get("rollout_pair_ids"):
            next_pair = int(batch["rollout_pair_ids"][0])
            out["pair_id"] = torch.tensor([next_pair], device=state_phys.device, dtype=torch.long)
            out["rollout_pair_ids"] = batch["rollout_pair_ids"][1:]
        if batch.get("rollout_phases") is not None:
            out["rollout_phases"] = batch["rollout_phases"][:, 1:]
    out["phase_next"] = float(context.get("phase_tp1", phase + phase_delta))
    out["phase_delta"] = phase_delta
    out["next_frame_id"] = context.get("frame_tp1", "")
    return out


def rebuild_next_batch_training(batch, state_phys, particle_field_phys, normalization, pushforward=False):
    """Training reconstruction preserves BPTT paths except VTK lookup, or accepts detached pushforward states."""
    return rebuild_next_batch(batch, state_phys, particle_field_phys, normalization, differentiable=not pushforward)


def rebuild_next_batch_inference(batch, state_phys, particle_field_phys, normalization):
    """Inference reconstruction runs under inference_mode and refreshes VTK geometry on CPU."""
    return rebuild_next_batch(batch, state_phys, particle_field_phys, normalization, differentiable=False)
