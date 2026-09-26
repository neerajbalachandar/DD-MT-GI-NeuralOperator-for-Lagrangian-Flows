import numpy as np

from gino.data.preprocessing_pipeline import match_particle_identities


def test_particle_correspondence_matches_identity_not_row_position():
    i0, i1, identities = match_particle_identities(
        np.asarray(["p7", "p2", "p9"]), np.asarray(["p9", "p7", "p2"]))
    np.testing.assert_array_equal(i0, [0, 1, 2])
    np.testing.assert_array_equal(i1, [1, 2, 0])
    np.testing.assert_array_equal(identities, ["p7", "p2", "p9"])


def test_particle_correspondence_drops_missing_ids_and_rejects_duplicates():
    i0, i1, identities = match_particle_identities(
        np.asarray(["p1", "p2", "p3"]), np.asarray(["p3", "p1"]))
    np.testing.assert_array_equal(i0, [0, 2])
    np.testing.assert_array_equal(i1, [1, 0])
    np.testing.assert_array_equal(identities, ["p1", "p3"])
    try:
        match_particle_identities(np.asarray(["p1", "p1"]), np.asarray(["p1"]))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate identities must be rejected")
