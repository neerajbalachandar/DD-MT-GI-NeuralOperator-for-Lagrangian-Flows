from pathlib import Path

from . import preprocessing_pipeline


# Historical Task-1/Task-2 sources are retained in the legacy pipeline below;
# current preprocessing reads the generated VPM cases from this single root.
DEFAULT_VPM_ROOT = Path("/media/dysco/New Volume/Neeraj/NeuralOp_FLOWVPM_Data")


def run_preprocessing(source_root=None, output_dir=None, *, max_pairs_per_case=120, device="auto"):
    """Build from generated VPM data, using a consecutive prefix of each case."""
    requested_root = Path(source_root or DEFAULT_VPM_ROOT).expanduser().resolve()
    output_path = Path(output_dir or preprocessing_pipeline.SCRIPT_DIR / "processed_data_vpm").expanduser().resolve()
    if all((requested_root / split).is_dir() for split in ("train", "test", "validate")):
        return preprocessing_pipeline.preprocess_generated_vpm_cases(
            requested_root, output_path, max_pairs_per_case=max_pairs_per_case, device=device)
    raise FileNotFoundError(
        f"Generated FLOWVPM data root {requested_root} must contain train/, validate/, and test/. "
        "Pass the mounted dataset directory with --source-root. Legacy Task-1/Task-2 data is not "
        "selected automatically; its historical entry point is run_legacy_preprocessing()."
    )


def run_legacy_preprocessing(source_root=None, output_dir=None, field_root=None):
    """Explicit historical Task-1/Task-2 path; current scripts/preprocess.py does not call it."""
    # These roots are intentionally opt-in to avoid silently preprocessing the
    # older airfoil dataset when the generated FLOWVPM mount is unavailable.
    task1_root = Path(source_root or "/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task1").expanduser().resolve()
    task2_root = Path(field_root or "/media/neerajc/New Volume/Neeraj/NeuralOp_Data/task2").expanduser().resolve()
    output_path = Path(output_dir or preprocessing_pipeline.SCRIPT_DIR / "processed_data_legacy").expanduser().resolve()
    if not task1_root.is_dir() or not task2_root.is_dir():
        raise FileNotFoundError(f"Legacy Task-1/Task-2 roots must exist: {task1_root} and {task2_root}")
    preprocessing_pipeline.RAW_ROOT_CANDIDATES = [task1_root]
    preprocessing_pipeline.FIELD_ROOT_CANDIDATES = [task2_root]
    preprocessing_pipeline.OUT_ROOT = output_path
    preprocessing_pipeline.MERGED_ROOT = output_path / "merged_frames"
    preprocessing_pipeline.main()
    return output_path / "particle_evolution_dataset.npz"
