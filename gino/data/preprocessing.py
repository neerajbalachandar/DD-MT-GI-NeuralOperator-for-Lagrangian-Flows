from pathlib import Path

from . import preprocessing_pipeline


DEFAULT_TASK1_ROOT = Path("/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task1")
DEFAULT_TASK2_ROOT = Path("/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task2")


def run_preprocessing(source_root=None, output_dir=None, field_root=None):
    """Build the canonical processed dataset from Task-1 particles and Task-2 fields."""
    task1_root = Path(source_root or DEFAULT_TASK1_ROOT).expanduser().resolve()
    task2_root = Path(field_root or DEFAULT_TASK2_ROOT).expanduser().resolve()
    output_path = Path(output_dir or preprocessing_pipeline.SCRIPT_DIR / "processed_data").expanduser().resolve()
    if not task1_root.is_dir():
        raise FileNotFoundError(f"Task-1 source directory does not exist: {task1_root}")
    if not task2_root.is_dir():
        raise FileNotFoundError(f"Task-2 source directory does not exist: {task2_root}")

    preprocessing_pipeline.RAW_ROOT_CANDIDATES = [task1_root]
    preprocessing_pipeline.FIELD_ROOT_CANDIDATES = [task2_root]
    preprocessing_pipeline.OUT_ROOT = output_path
    preprocessing_pipeline.MERGED_ROOT = output_path / "merged_frames"
    preprocessing_pipeline.main()
    return output_path / "particle_evolution_dataset.npz"


# Kept as a compatibility alias for callers from earlier package revisions.
run_legacy_preprocessing = run_preprocessing
