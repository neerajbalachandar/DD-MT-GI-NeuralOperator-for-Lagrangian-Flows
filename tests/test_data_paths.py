from pathlib import Path

from gino.data import preprocessing


def test_default_generated_vpm_root_points_to_repo_training_data():
    repo_root = Path(__file__).resolve().parents[1]
    expected_root = repo_root / "datagen_code" / "training_data"

    assert preprocessing.DEFAULT_VPM_ROOT == expected_root
    assert expected_root.is_dir()
    assert (expected_root / "train").is_dir()
    assert (expected_root / "test").is_dir()
    assert (expected_root / "validate").is_dir()
