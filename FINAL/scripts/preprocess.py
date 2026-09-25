import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gino.data.preprocessing import run_legacy_preprocessing


def main():
    parser = argparse.ArgumentParser(description="Build the canonical Task-1/Task-2 pair dataset.")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--field-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(run_legacy_preprocessing(args.source_root, args.output_dir, args.field_root))


if __name__ == "__main__":
    main()
