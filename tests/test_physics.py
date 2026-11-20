import torch
from gino.dynamics.physics import physical_transition


def test_euler_vpm_physical_transition():
    state = torch.tensor([[[1., 2., 3., 1., 2., 3., 0.5]]])
    velocity = torch.ones((1, 1, 3))
    grad = torch.eye(3).reshape(1, 1, 3, 3)
    result = physical_transition(state, velocity, grad, 0.1)
    torch.testing.assert_close(result[..., :3], state[..., :3] + 0.1)
    torch.testing.assert_close(result[..., 3:6], state[..., 3:6] * 1.1)
    torch.testing.assert_close(result[..., 6], state[..., 6])


def test_stretching_scheme_transposed_is_default_and_matches_flowvpm():
    # grad[i, j] = du_i/dx_j with a single non-symmetric entry du_x/dy = 2.
    grad = torch.zeros(1, 1, 3, 3)
    grad[..., 0, 1] = 2.0
    state = torch.zeros(1, 1, 7)
    state[..., 3] = 1.0   # Gamma_x
    state[..., 4] = 1.0   # Gamma_y
    velocity = torch.zeros(1, 1, 3)
    # FLOWVPM transposed=true: dGamma_i = sum_j du_j/dx_i Gamma_j -> only dGamma_y = du_x/dy * Gamma_x = 2.
    transposed = physical_transition(state, velocity, grad, 0.5)
    torch.testing.assert_close(transposed[0, 0, 3:6], torch.tensor([1.0, 2.0, 0.0]))
    torch.testing.assert_close(
        physical_transition(state, velocity, grad, 0.5, scheme="transposed"), transposed)
    # Classic: dGamma_i = sum_j du_i/dx_j Gamma_j -> only dGamma_x = du_x/dy * Gamma_y = 2.
    classic = physical_transition(state, velocity, grad, 0.5, scheme="classic")
    torch.testing.assert_close(classic[0, 0, 3:6], torch.tensor([2.0, 1.0, 0.0]))


def test_stretching_scheme_rejects_unknown_name():
    import pytest
    with pytest.raises(ValueError):
        physical_transition(torch.zeros(1, 1, 7), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3, 3), 0.1, scheme="bogus")
