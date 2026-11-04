import numpy as np
import pytest

from gino.data.dataset import (
    sequence_level_split,
    assert_no_sequence_leakage,
    ensure_legacy_input_feature_columns,
)
from gino.data.geometry import particle_geometry_features
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


def test_legacy_dataset_missing_delta_t_is_backfilled():
    data = tiny_data()
    data["feature_names"] = np.asarray(["x", "y", "z"], dtype=object)
    data["inputs_t"] = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    data["in_mean"] = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    data["in_std"] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    data["pair_ranges"] = np.asarray([
        ("case-a", "0", "1", 0, 1, 1),
        ("case-a", "1", "2", 1, 2, 1),
    ], dtype=object)
    data["pair_contexts"] = np.asarray([
        {"dt": 0.25},
        {"dt": 0.50},
    ], dtype=object)

    result = ensure_legacy_input_feature_columns(data, ["x", "y", "z", "delta_t"])
    assert np.asarray(result["feature_names"]).tolist() == ["x", "y", "z", "delta_t"]
    np.testing.assert_allclose(result["inputs_t"][:, -1], [0.25, 0.50])


def test_missing_vtk_geometry_falls_back_to_zero_channels():
    xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    features = particle_geometry_features(xyz, "/definitely/missing/geometry.vtk", case="case-a")

    for key in ("geom_dist", "geom_nx", "geom_ny", "geom_nz", "geom_body_near"):
        np.testing.assert_array_equal(features[key], np.zeros(2, dtype=np.float64))
