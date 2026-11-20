"""Export raw FLOWVPM vortex-ring cases into lossless per-case archives.

Usage:
    python scripts/build_vortex_dataset.py \
        --raw-root datagen_code/training_data --out processed_data/vortex_rings_v2
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from gino.data.vortex_raw import export_cases  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=str(HERE / "datagen_code/training_data"))
    parser.add_argument("--out", default=str(HERE / "processed_data/vortex_rings_v2"))
    parser.add_argument("--family", default="vortexrings")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--case", action="append", default=None, help="export only these case names")
    args = parser.parse_args()
    manifest = export_cases(args.raw_root, args.out, args.family, overwrite=args.overwrite, include=args.case)
    print(f"[export] manifest: {manifest}")


if __name__ == "__main__":
    main()
