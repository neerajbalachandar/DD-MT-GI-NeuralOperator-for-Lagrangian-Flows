import numpy as np
import pytest

from gino.evaluation import field as field_module


def test_native_field_comparison_rejects_nonidentical_coordinates(monkeypatch):
    coords = np.array([[0., 0., 0.], [1., 0., 0.]])
    monkeypatch.setattr(field_module, "read_task2_field", lambda path: (coords, np.zeros((2, 3))))
    with pytest.raises(ValueError, match="identical coordinates"):
        field_module.evaluate_native_field(coords + 1e-3, np.zeros((2, 3)), "native.h5")
