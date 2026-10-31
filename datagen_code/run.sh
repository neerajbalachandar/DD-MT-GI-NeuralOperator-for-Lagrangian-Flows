#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Usage: ./run.sh [all|vortexrings|roundjet]
DATASET="${1:-${DATASET:-all}}"
case "$DATASET" in
    all|vortexrings|roundjet) ;;
    *)
        echo "Usage: $0 [all|vortexrings|roundjet]" >&2
        exit 2
        ;;
esac

# datagen_code is one level below FLOWUnsteady; use FLOWUnsteady/Project.toml by default.
JULIA_PROJECT="${JULIA_PROJECT:-$SCRIPT_DIR/..}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/training_data}"
GRID_POINTS="${GRID_POINTS:-32}"
SAVE_EVERY="${SAVE_EVERY:-10}"
export OUTPUT_ROOT GRID_POINTS SAVE_EVERY

command -v julia >/dev/null 2>&1 || {
    echo "Julia was not found on PATH." >&2
    exit 127
}

exec julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/training_sweep.jl" "$DATASET"
