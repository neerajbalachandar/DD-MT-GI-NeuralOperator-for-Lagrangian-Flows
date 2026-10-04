import numpy as np
import pytest

from gino.data.preprocessing_pipeline import (
    _aligned_rollout_truth,
    match_task2_xmf_time,
    _require_scalar,
    _validate_structured_node_order,
    validate_task2_xmf_layout,
)


def test_task2_xmf_declares_65_cubed_nodal_data(tmp_path):
    h5_path = tmp_path / "field.20.h5"
    xmf_path = tmp_path / "field.20.xmf"
    xmf_path.write_text(
        '<Xdmf><Domain><Grid>'
        '<Geometry><DataItem Dimensions="3 274625">field.20.h5:nodes</DataItem></Geometry>'
        '<Attribute Name="U" Center="Node"><DataItem Dimensions="65 65 65 3">'
        'field.20.h5:U</DataItem></Attribute>'
        '</Grid></Domain></Xdmf>'
    )
    validate_task2_xmf_layout(xmf_path, h5_path)


def test_task2_xmf_rejects_wrong_nodal_count(tmp_path):
    h5_path = tmp_path / "field.20.h5"
    xmf_path = tmp_path / "field.20.xmf"
    xmf_path.write_text(
        '<Xdmf><Domain><Grid>'
        '<Geometry><DataItem Dimensions="3 8">field.20.h5:nodes</DataItem></Geometry>'
        '<Attribute Name="U" Center="Node"><DataItem Dimensions="2 2 2 3">'
        'field.20.h5:U</DataItem></Attribute>'
        '</Grid></Domain></Xdmf>'
    )
    with pytest.raises(ValueError, match="expected nodes and nodal U"):
        validate_task2_xmf_layout(xmf_path, h5_path)


def test_task2_field_matching_uses_xmf_time_not_filename_index(tmp_path):
    path = tmp_path / "static_airfoil_fdom.20.h5"
    assert match_task2_xmf_time([(20.0, path)], 20.0) == (20.0, path)
    assert match_task2_xmf_time([(20.0, path)], 19.0) is None


def test_structured_xmf_nodes_follow_velocity_tensor_order():
    axes = (np.arange(3.0), np.arange(4.0), np.arange(5.0))
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    _validate_structured_node_order(coords, (3, 4, 5), "synthetic.h5")
    shuffled = coords[np.random.default_rng(8).permutation(len(coords))]
    with pytest.raises(ValueError, match="does not reshape to a rectilinear grid"):
        _validate_structured_node_order(shuffled, (3, 4, 5), "shuffled.h5")


def test_particle_scalar_lengths_must_match_particle_count():
    np.testing.assert_array_equal(_require_scalar(np.arange(3), 3, "sigma"), [0, 1, 2])
    with pytest.raises(ValueError, match="exactly 3 values"):
        _require_scalar(np.arange(2), 3, "sigma")
    with pytest.raises(ValueError, match="exactly 3 values"):
        _require_scalar(np.arange(4), 3, "static")


def test_rollout_truth_uses_persistent_ids_not_row_positions():
    frames = [
        {"particle_ids": np.asarray(["a", "b"]),
         "state": {"x": np.asarray([1., 2.]), "y": np.zeros(2), "z": np.zeros(2),
                   "Gamma_x": np.zeros(2), "Gamma_y": np.zeros(2), "Gamma_z": np.zeros(2), "sigma": np.ones(2)}},
        {"particle_ids": np.asarray(["b", "a"]),
         "state": {"x": np.asarray([20., 10.]), "y": np.zeros(2), "z": np.zeros(2),
                   "Gamma_x": np.zeros(2), "Gamma_y": np.zeros(2), "Gamma_z": np.zeros(2), "sigma": np.ones(2)}},
    ]
    states, ids = _aligned_rollout_truth(frames)
    assert ids.tolist() == ["a", "b"]
    np.testing.assert_array_equal(states[:, :, 0], [[1., 2.], [10., 20.]])
