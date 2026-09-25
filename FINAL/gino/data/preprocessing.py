from pathlib import Path
import sys


def run_legacy_preprocessing(source_root=None, output_dir=None, field_root=None):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import process_data
    if source_root is not None:
        process_data.RAW_ROOT_CANDIDATES = [Path(source_root).expanduser().resolve()]
    if field_root is not None:
        process_data.FIELD_ROOT_CANDIDATES = [Path(field_root).expanduser().resolve()]
    if output_dir is not None:
        process_data.OUT_ROOT = Path(output_dir).expanduser().resolve()
        process_data.MERGED_ROOT = process_data.OUT_ROOT / "merged_frames"
    process_data.main()
    return process_data.OUT_ROOT / "particle_evolution_dataset.npz"
