import torch


def random_walk_noise(states, initial_std: float, walk_std: float, generator=None):
    """Pre-generate normalized-feature noise [time, ...] for one rollout sequence."""
    if states.ndim < 2:
        raise ValueError("Expected time and feature dimensions")
    if states.shape[0] == 0:
        return states.clone()
    eps = torch.randn(states[0].shape, dtype=states.dtype, device=states.device,
                      generator=generator) * float(initial_std)
    output = [eps]
    for _ in range(1, states.shape[0]):
        increment = torch.randn(eps.shape, dtype=eps.dtype, device=eps.device,
                                generator=generator) * float(walk_std)
        eps = eps + increment
        output.append(eps)
    return torch.stack(output, dim=0)


def perturb_flow_inputs(batch, noise, feature_indices):
    """Add one pre-generated noise slice in normalized feature space."""
    if not feature_indices:
        return batch["x"], noise
    x = batch["x"].clone()
    if noise is None:
        raise ValueError("A pre-generated normalized-space noise slice is required")
    x[..., feature_indices] = x[..., feature_indices] + noise
    return x, noise
