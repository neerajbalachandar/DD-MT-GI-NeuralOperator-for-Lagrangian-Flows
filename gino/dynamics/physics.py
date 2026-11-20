import torch


# grad_velocity is row-major: grad[..., i, j] = du_i / dx_j.
#   classic    : dGamma_i/dt = (Gamma . grad) u      = sum_j du_i/dx_j Gamma_j  -> einsum "...ij,...j->...i"
#   transposed : dGamma_i/dt = (Gamma . grad^T) u    = sum_j du_j/dx_i Gamma_j  -> einsum "...ji,...j->...i"
# The FLOWVPM simulations used to generate the data run with transposed=true,
# so the transposed scheme is the default (the previous implementation always
# used the classic scheme, which is inconsistent with the reference solver).
_STRETCHING_EINSUM = {
    "transposed": "...ji,...j->...i",
    "classic": "...ij,...j->...i",
}


def physical_transition(state, velocity, grad_velocity, dt, scheme="transposed"):
    """Explicit Euler VPM base step for position and circulation.

    ``scheme`` selects the vortex-stretching formulation ('transposed' matches
    FLOWVPM's transposed=true; 'classic' is the non-transposed form).
    """
    if scheme not in _STRETCHING_EINSUM:
        raise ValueError(f"Unknown stretching scheme {scheme!r}; expected one of {sorted(_STRETCHING_EINSUM)}")
    next_state = state.clone()
    next_state[..., :3] = state[..., :3] + dt * velocity
    next_state[..., 3:6] = state[..., 3:6] + dt * torch.einsum(
        _STRETCHING_EINSUM[scheme], grad_velocity, state[..., 3:6])
    return next_state
