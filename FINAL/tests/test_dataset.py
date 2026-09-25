import numpy as np
import pytest

from gino.data.dataset import sequence_level_split, assert_no_sequence_leakage
from gino.data.normalization import NormalizationStats


def tiny_data():
    cases = ["a", "b", "c", "test"]
    ranges, contexts = [], []
    for i, case in enumerate(cases):
        ranges.append((case, "0", "1", i * 2, i * 2 + 2, 2))
        contexts.append({"case": case})
    x = np.arange(4 * 2 * 3, dtype=np.float32).reshape(8, 3)
    return {"pair_ranges": np.asarray(ranges, dtype=object), "pair_contexts": np.asarray(contexts, dtype=object),
            "train_pair_ids": np.array([0, 1, 2]), "test_cases": np.array(["test"]),
            "inputs_t": x, "feature_names": np.array(["x", "y", "z"]), "in_mean": x[:6].mean(0),
            "in_std": x[:6].std(0), "targets_residual": x.copy(), "targets_velocity_field": np.ones((4, 2, 1)),
            "field_query_mask": np.ones((4, 2), dtype=bool), "query_coords": np.zeros((4, 2, 3)),
            "field_mean": np.array([1.]), "field_std": np.array([1.]), "coord_min": np.zeros(3), "coord_span": np.ones(3)}


def test_sequence_split_has_no_case_leakage():
    data = tiny_data()
    splits = sequence_level_split(data, seed=7)
    assert_no_sequence_leakage(data, splits)
    assert splits["test_cases"] == ["test"]


def test_train_normalization_uses_selected_training_pairs_only():
    data = tiny_data()
    stats = NormalizationStats.fit_train_sequences(data, [0, 1], [0, 1, 2])
    np.testing.assert_allclose(stats.input_mean, data["inputs_t"][:4].mean(0))
    assert not np.allclose(stats.input_mean, data["inputs_t"].mean(0))
