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
