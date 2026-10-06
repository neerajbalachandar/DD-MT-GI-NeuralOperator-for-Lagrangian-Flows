import numpy as np

from gino.data.preprocessing_pipeline import _to_s32


def test_s32_particle_id_roundtrip():
    original = np.array(["a", "bb", "ccc"])
    encoded = _to_s32(original)

    assert encoded.dtype == np.dtype("S32")
    np.testing.assert_array_equal(np.char.decode(encoded, "utf-8"), original)
