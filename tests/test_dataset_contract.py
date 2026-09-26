import numpy as np

from gino.data.dataset import EvolutionDataset
from gino.data.normalization import NormalizationStats


def _toy_data():
    names = np.asarray(["x", "y", "z", "u_x", "Gamma_x", "Gamma_y", "Gamma_z", "sigma"], dtype=object)
    inputs = np.zeros((4, len(names)), dtype=np.float32)
    inputs[:, 3] = [2, 2, 100, 100]
    states = np.zeros((4, 7), dtype=np.float32)
    states[:, 0] = [1, 1, 2, 2]
    return {
        "feature_names": names,
        "pair_ranges": np.asarray([["c", "0", "1", 0, 2, 2], ["c", "1", "2", 2, 4, 2]], dtype=object),
        "inputs_t": inputs,
        "targets_next_state": states,
        "targets_residual": np.zeros((4, 10), dtype=np.float32),
        "targets_delta": np.zeros((4, 10), dtype=np.float32),
        "pair_contexts": np.asarray([
            {"case": "c", "frame_t": "0", "frame_tp1": "1", "phase_t": 0., "phase_tp1": .5,
             "phase_delta": .5, "dt": .1, "vtk_path": "frame1.vtk"},
            {"case": "c", "frame_t": "1", "frame_tp1": "2", "phase_t": .5, "phase_tp1": 1.,
             "phase_delta": .5, "dt": .1, "vtk_path": "frame2.vtk"}], dtype=object),
        "field_query_mask": np.ones((2, 2), dtype=bool),
        "query_coords": np.zeros((2, 2, 3), dtype=np.float32),
        "targets_velocity_field": np.zeros((2, 2, 12), dtype=np.float32),
        "in_mean": np.zeros(len(names), dtype=np.float32), "in_std": np.ones(len(names), dtype=np.float32),
        "out_mean": np.zeros(10, dtype=np.float32), "out_std": np.ones(10, dtype=np.float32),
        "field_mean": np.zeros(12, dtype=np.float32), "field_std": np.ones(12, dtype=np.float32),
        "coord_min": np.zeros(3, dtype=np.float32), "coord_span": np.ones(3, dtype=np.float32),
    }


def test_train_only_statistics_and_index_zero_is_one_step_target():
    data = _toy_data()
    stats = NormalizationStats.fit_train_sequences(data, [0], [3])
    np.testing.assert_allclose(stats.input_mean, [2.])
    dataset = EvolutionDataset(data, [0], ["u_x"], [], max_particles=8, max_queries=8,
                               rollout_horizon=2, normalization=stats)
    sample = dataset[0]
    np.testing.assert_allclose(sample["rollout_state_targets"][0], sample["one_step_target"])
    assert sample["rollout_state_targets"].shape == (2, 2, 7)
    assert sample["rollout_time_indices"].tolist() == [1, 2]
    assert sample["rollout_phases"].tolist() == [0.5, 1.0]
    assert sample["rollout_contexts"][0]["vtk_path"] == "frame2.vtk"


def test_rollout_rejects_unverified_legacy_row_correspondence():
    data = _toy_data()
    stats = NormalizationStats.fit_train_sequences(data, [0], [3])
    dataset = EvolutionDataset(data, [0], ["u_x"], [], max_particles=8, max_queries=8,
        rollout_horizon=2, normalization=stats, require_verified_correspondence=True)
    try:
        dataset[0]
    except ValueError as error:
        assert "no verified physical particle correspondence" in str(error)
    else:
        raise AssertionError("unverified row ordering must not support rollout evaluation")
