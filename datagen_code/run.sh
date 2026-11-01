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
JULIA="${JULIA:-julia}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/training_data_temporal4_32}"
# Mid/high-fidelity defaults: resolve fields on a 32^3 mesh and save one in
# every four simulation steps. Particle and field snapshots share this cadence.
# Set CASE_LIMIT to cap each dataset sweep on resource-constrained runs.
GRID_POINTS="${GRID_POINTS:-32}"
SAVE_EVERY="${SAVE_EVERY:-4}"
CASE_LIMIT="${CASE_LIMIT:-0}"
VORTEX_STEPS="${VORTEX_STEPS:-1000}"
VORTEX_RTOT="${VORTEX_RTOT:-10}"
ROUNDJET_STEPS_PER_D="${ROUNDJET_STEPS_PER_D:-100}"
ROUNDJET_TRAVEL_D="${ROUNDJET_TRAVEL_D:-10}"
ROUNDJET_RE="${ROUNDJET_RE:-10000}"
ROUNDJET_RBF_ITMAX="${ROUNDJET_RBF_ITMAX:-200}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export OUTPUT_ROOT GRID_POINTS SAVE_EVERY CASE_LIMIT
export VORTEX_STEPS VORTEX_RTOT ROUNDJET_STEPS_PER_D ROUNDJET_TRAVEL_D ROUNDJET_RE ROUNDJET_RBF_ITMAX

[[ -f "$JULIA_PROJECT/Project.toml" ]] || {
    echo "Julia project not found: $JULIA_PROJECT/Project.toml" >&2
    echo "Set JULIA_PROJECT to the FLOWUnsteady environment directory." >&2
    exit 2
}

command -v "$JULIA" >/dev/null 2>&1 || {
    echo "Julia executable was not found: $JULIA" >&2
    exit 127
}

# Ensure the active project environment matches the checked-in dependency set.
"$JULIA" --project="$JULIA_PROJECT" -e 'using Pkg; Pkg.instantiate()' >/dev/null

exec "$JULIA" --project="$JULIA_PROJECT" "$SCRIPT_DIR/training_sweep.jl" "$DATASET"
