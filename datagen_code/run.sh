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

# The Julia project lives in the parent directory of the repository checkout,
# not in the repository root itself. Keep the default in sync with that layout.
FLOW_UNSTEADY_ROOT="${FLOW_UNSTEADY_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
JULIA_PROJECT="${JULIA_PROJECT:-$FLOW_UNSTEADY_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/training_data}"
# Keep default local runs on a lower-footprint configuration. The full sweep is
# too large for memory-constrained machines, so default to a single representative
# case unless the user explicitly opts into the broader run.
GRID_POINTS="${GRID_POINTS:-8}"
SAVE_EVERY="${SAVE_EVERY:-50}"
CASE_LIMIT="${CASE_LIMIT:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export OUTPUT_ROOT GRID_POINTS SAVE_EVERY CASE_LIMIT

command -v julia >/dev/null 2>&1 || {
    echo "Julia was not found on PATH." >&2
    exit 127
}

# Ensure the active project environment matches the checked-in dependency set.
julia --project="$JULIA_PROJECT" -e 'using Pkg; Pkg.instantiate()' >/dev/null

exec julia --project="$JULIA_PROJECT" "$SCRIPT_DIR/training_sweep.jl" "$DATASET"
