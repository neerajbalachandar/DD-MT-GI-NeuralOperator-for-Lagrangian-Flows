import torch


def random_walk_noise(states, initial_std: float, walk_std: float, generator=None):
    """Temporally correlated additive noise for a (time, ..., channel) sequence."""
    if states.ndim < 2:
        raise ValueError("Expected time and feature dimensions")
    eps = torch.randn(states.shape[1:], dtype=states.dtype, device=states.device, generator=generator) * float(initial_std)
    output = []
    for t in range(states.shape[0]):
        if t:
            eps = eps + torch.randn(eps.shape, dtype=eps.dtype, device=eps.device, generator=generator) * float(walk_std)
        output.append(eps)
    return torch.stack(output)
