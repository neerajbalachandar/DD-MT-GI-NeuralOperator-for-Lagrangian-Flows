import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser(description="Preprocess generated FLOWVPM particle and field data.")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Generated root containing train/, validate/, and test/ (default: configured FLOWVPM data root).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-pairs-per-case", type=int, default=0,
                        help="Optional cap on consecutive transitions per case; default 0 processes every available pair.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Use CUDA for field-gradient preprocessing when available.")
    args = parser.parse_args()
    from gino.data.preprocessing import run_preprocessing

    max_pairs = args.max_pairs_per_case or None
    print(run_preprocessing(args.source_root, args.output_dir,
                            max_pairs_per_case=max_pairs,
                            device=args.device))


if __name__ == "__main__":
    main()
