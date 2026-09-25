import torch


def physical_transition(state, velocity, grad_velocity, dt):
    """Existing explicit Euler VPM base step for position and circulation."""
    next_state = state.clone()
    next_state[..., :3] = state[..., :3] + dt * velocity
    next_state[..., 3:6] = state[..., 3:6] + dt * torch.einsum("...ij,...j->...i", grad_velocity, state[..., 3:6])
    return next_state
