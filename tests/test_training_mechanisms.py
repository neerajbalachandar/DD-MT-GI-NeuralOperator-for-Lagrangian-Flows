import torch
import pytest

from gino.training.losses import (normalized_rollout_loss, grouped_field_loss,
                                  uncertainty_weighted_mse, uncertainty_weighted_sum)
from gino.training.noise import perturb_flow_inputs, random_walk_noise, noise_for_rollout_input
from gino.training.scheduled_sampling import (choose_predicted_flow_inputs,
    scheduled_sampling_probability, teacher_input_at)
from scripts.train import select_configured_input_features
import numpy as np
from gino.training.losses import normalized_physical_state_loss


def test_random_walk_noise_shape_and_zero_variance():
    sequence = torch.zeros(4, 1, 3, 5)
    noise = random_walk_noise(sequence, initial_std=0.0, walk_std=0.0)
    assert noise.shape == sequence.shape
    assert torch.count_nonzero(noise) == 0


def test_random_walk_noise_is_temporally_correlated_and_indexed_by_input_time():
    torch.manual_seed(5)
    sequence = torch.zeros(4, 1, 1, 1)
    noise = random_walk_noise(sequence, initial_std=0.2, walk_std=0.05)
    assert noise_for_rollout_input(noise, 0).data_ptr() == noise[0].data_ptr()
    assert noise_for_rollout_input(noise, 2).data_ptr() == noise[2].data_ptr()
    increments = noise[1:] - noise[:-1]
    assert torch.isfinite(increments).all()
    assert torch.count_nonzero(increments) == 3
    # A fixed seed makes the indexed input sequence reproducible across rollout callbacks.
    torch.manual_seed(5)
    repeated = random_walk_noise(sequence, initial_std=0.2, walk_std=0.05)
    torch.testing.assert_close(noise, repeated)


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


def test_teacher_input_selection_advances_with_rollout_timestep():
    teachers = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    torch.testing.assert_close(teacher_input_at(teachers, 0), teachers[:, 0])
    torch.testing.assert_close(teacher_input_at(teachers, 2), teachers[:, 2])


def test_normalized_rollout_loss_scales_each_state_component():
    predicted = torch.tensor([[[[2.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0]]]])
    loss = normalized_rollout_loss(predicted, target, mean=[0.0, 0.0], std=[1.0, 2.0])
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_grouped_field_loss_slices_canonical_channels_with_independent_mse():
    target = torch.ones((1, 4, 15))
    prediction = target.clone()
    prediction[..., :3] += 1.0
    prediction[..., 3:12] += 2.0
    prediction[..., 12:15] += 3.0
    aggregate, groups = grouped_field_loss(prediction, target)
    torch.testing.assert_close(groups["velocity"], torch.tensor(1.0))
    torch.testing.assert_close(groups["gradient"], torch.tensor(4.0))
    torch.testing.assert_close(groups["vorticity"], torch.tensor(9.0))
    torch.testing.assert_close(aggregate, torch.tensor(14.0))


def test_grouped_field_mse_exact_constant_and_query_count_independent():
    target = torch.arange(30, dtype=torch.float32).reshape(1, 2, 15)
    aggregate, groups = grouped_field_loss(target, target)
    assert aggregate.item() == 0.0
    assert all(value.item() == 0.0 for value in groups.values())
    for n_queries in (1, 4, 19):
        prediction = torch.ones((2, n_queries, 15))
        reference = torch.zeros_like(prediction)
        total, pieces = grouped_field_loss(prediction, reference)
        assert set(pieces) == {"velocity", "gradient", "vorticity"}
        assert all(value.item() == 1.0 for value in pieces.values())
        assert total.item() == 3.0


def test_uncertainty_weighted_losses_have_gradients_for_every_task_variance():
    names = ("physical_state", "residual", "field_velocity", "field_gradient", "field_vorticity")
    variances = {name: torch.nn.Parameter(torch.zeros(())) for name in names}
    losses = {name: torch.tensor(2.0, requires_grad=True) for name in names}
    result = uncertainty_weighted_sum(losses, variances)
    result.backward()
    for parameter in variances.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
    torch.testing.assert_close(uncertainty_weighted_mse(torch.tensor(2.), torch.tensor(0.)), torch.tensor(1.))


def test_training_preserves_explicit_state_only_feature_selection():
    available = np.asarray(["x", "y", "z", "u_x", "gradU_xx", "Gamma_x", "Gamma_y", "Gamma_z", "sigma"])
    requested = ["Gamma_x", "Gamma_y", "Gamma_z", "sigma"]
    assert select_configured_input_features({"feature_names": available}, requested) == requested
    with pytest.raises(ValueError, match="missing"):
        select_configured_input_features({"feature_names": available}, ["Gamma_x", "u_z"])


def test_physical_state_loss_normalizes_both_prediction_and_target():
    mean = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
    std = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
    target = mean.view(1, 1, 7).clone()
    predicted = target.clone()
    predicted[..., 0] += 1.0
    loss = normalized_physical_state_loss(predicted, target, mean, std)
    torch.testing.assert_close(loss, torch.tensor(1.0 / 7.0))


def test_grouped_field_loss_rejects_invalid_targets_instead_of_masking():
    target = torch.ones((1, 2, 15))
    target[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinite"):
        grouped_field_loss(torch.zeros_like(target), target)
