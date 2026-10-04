import numpy as np

from visualization.fields import plot_native_ux_comparison, plot_normalized_ux_profiles
from visualization.temporal import (
    plot_one_step_loss_vs_time,
    plot_rollout_error_by_step,
    plot_rollout_snapshots,
)


def test_requested_temporal_plots_write_outputs(tmp_path):
    one_step = [
        {"case": "a", "phase": 0.2, "position_relative_l2": 0.1,
         "field_phase": 0.2, "field_relative_l2": 0.2},
        {"case": "b", "phase": 0.2, "position_relative_l2": 0.2,
         "field_phase": 0.2, "field_relative_l2": 0.3},
    ]
    rollout = [
        {"case": "a", "horizon": 1, "position_relative_l2": 0.1},
        {"case": "b", "horizon": 1, "position_relative_l2": 0.2},
        {"case": "a", "horizon": 2, "position_relative_l2": 0.3},
        {"case": "b", "horizon": 2, "position_relative_l2": 0.4},
    ]
    plot_one_step_loss_vs_time(one_step, tmp_path / "one_step.png")
    plot_rollout_error_by_step(rollout, tmp_path / "rollout.png")
    assert (tmp_path / "one_step.png").is_file()
    assert (tmp_path / "rollout.png").is_file()


def test_native_field_and_profile_plots_use_matching_velocity_grids(tmp_path):
    x, z = np.linspace(0, 2, 8), np.linspace(-1, 1, 7)
    xx, zz = np.meshgrid(x, z)
    ux = np.exp(-((xx - 0.8) ** 2 + zz ** 2))
    true_velocity = np.stack((ux, 0.2 * ux, 0.1 * ux), axis=-1)
    prediction = true_velocity * 0.95
    plot_native_ux_comparison(x, z, ux, prediction[..., 0], tmp_path / "field.png", phase=0.5)
    plot_normalized_ux_profiles(x, z, true_velocity, prediction, tmp_path / "profiles.png",
                                freestream_speed=10.0, chord=1.0, leading_edge_x=0.0)
    assert (tmp_path / "field.png").is_file()
    assert (tmp_path / "profiles.png").is_file()


def test_rollout_snapshots_plot_true_and_predicted_particles(tmp_path):
    state = np.zeros((12, 7), dtype=np.float32)
    state[:, 0] = np.linspace(0, 1, len(state))
    state[:, 2] = np.linspace(-0.2, 0.2, len(state))
    state[:, 3] = 1.0
    state[:, 6] = 0.1
    plot_rollout_snapshots({1: {"truth": state, "prediction": state * 1.01,
                               "physical_time": 0.1}}, tmp_path / "particles.png")
    assert (tmp_path / "particles.png").is_file()
