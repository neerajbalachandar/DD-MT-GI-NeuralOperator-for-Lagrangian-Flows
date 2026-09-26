import torch

from gino.training.losses import normalized_rollout_loss
from gino.training.noise import perturb_flow_inputs, random_walk_noise
from gino.training.scheduled_sampling import choose_predicted_flow_inputs, scheduled_sampling_probability


def test_random_walk_noise_shape_and_zero_variance():
    sequence = torch.zeros(4, 1, 3, 5)
    noise = random_walk_noise(sequence, initial_std=0.0, walk_std=0.0)
    assert noise.shape == sequence.shape
    assert torch.count_nonzero(noise) == 0


def test_zero_gns_noise_preserves_baseline_input():
    batch = {"x": torch.randn(1, 3, 4)}
    original = batch["x"].clone()
    noise = torch.zeros(1, 3, 2)
    changed, _ = perturb_flow_inputs(batch, noise, [0, 2])
    torch.testing.assert_close(changed, original)
    assert torch.count_nonzero(noise) == 0


def test_scheduled_sampling_changes_only_velocity_and_gradient_features():
    batch = {"x": torch.zeros(1, 2, 3), "feature_names": ["u_x", "gradU_yz", "sigma"],
             "input_mean": torch.zeros(3), "input_std": torch.ones(3)}
    velocity = torch.ones(1, 2, 3)
    gradient = torch.ones(1, 2, 3, 3) * 2.0
    changed, used = choose_predicted_flow_inputs(batch, velocity, gradient, probability=1.0)
    assert used
    torch.testing.assert_close(changed[..., 0], torch.ones(1, 2))
    torch.testing.assert_close(changed[..., 1], torch.full((1, 2), 2.0))
    torch.testing.assert_close(changed[..., 2], torch.zeros(1, 2))
    assert scheduled_sampling_probability(1, 0.3) == 0.0
    assert scheduled_sampling_probability(31, 0.3) == 0.3


def test_scheduled_sampling_uses_one_choice_and_teacher_flow_only():
    batch = {"x": torch.zeros(1, 2, 3), "feature_names": ["u_x", "gradU_yz", "sigma"],
             "input_mean": torch.zeros(3), "input_std": torch.ones(3)}
    teacher = torch.tensor([[[4.0, 5.0, 9.0], [4.0, 5.0, 9.0]]])
    velocity = torch.ones(1, 2, 3)
    gradient = torch.ones(1, 2, 3, 3) * 2
    changed, used = choose_predicted_flow_inputs(batch, velocity, gradient, 0.0, teacher_x=teacher)
    assert not used
    torch.testing.assert_close(changed[..., :2], teacher[..., :2])
    torch.testing.assert_close(changed[..., 2], torch.zeros(1, 2))


def test_normalized_rollout_loss_scales_each_state_component():
    predicted = torch.tensor([[[[2.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0]]]])
    loss = normalized_rollout_loss(predicted, target, mean=[0.0, 0.0], std=[1.0, 2.0])
    torch.testing.assert_close(loss, torch.tensor(1.0))
