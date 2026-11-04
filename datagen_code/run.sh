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
OUTPUT_ROOT="${OUTPUT_ROOT:-/hpc/home/neerajc/FLOWUnsteady/Flow-reconstruction-in-VPM-using-FNO/datagen_code/training_data}" 
# Save about 100 transitions per 1000-step standard case. Resume skips complete
# cases and removes/restarts only incomplete case directories.
GRID_POINTS="${GRID_POINTS:-32}"
SAVE_EVERY="${SAVE_EVERY:-4}"
# FLOWVPM otherwise reports only every 100 steps, which can look like a stall
# even though particle snapshots continue to be written every SAVE_EVERY steps.
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
CASE_LIMIT="${CASE_LIMIT:-0}"
RESUME="${RESUME:-1}"
VORTEX_STEPS="${VORTEX_STEPS:-720}"
VORTEX_RTOT="${VORTEX_RTOT:-7.2}"
ROUNDJET_STEPS_PER_D="${ROUNDJET_STEPS_PER_D:-50}"
ROUNDJET_TRAVEL_D="${ROUNDJET_TRAVEL_D:-7.2}"
ROUNDJET_PULSE_D="${ROUNDJET_PULSE_D:-0.10}"
ROUNDJET_RE="${ROUNDJET_RE:-10000}"
# The full CG RBF fit costs several minutes per operator evaluation at the
# planned roundjet resolutions. Use the analytic annulus strengths by default;
# set ROUNDJET_RBF=1 to opt into the iterative fit.
ROUNDJET_RBF="${ROUNDJET_RBF:-0}"
ROUNDJET_RBF_ITMAX="${ROUNDJET_RBF_ITMAX:-200}"
ROUNDJET_RBF_TIMEOUT_SECONDS="${ROUNDJET_RBF_TIMEOUT_SECONDS:-1800}"
# Keep parallel execution while leaving headroom for the FMM tree and temporary
# grid particles. Override JULIA_NUM_THREADS on machines with more memory.
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"
export OUTPUT_ROOT GRID_POINTS SAVE_EVERY PROGRESS_EVERY CASE_LIMIT RESUME
export VORTEX_STEPS VORTEX_RTOT ROUNDJET_STEPS_PER_D ROUNDJET_TRAVEL_D ROUNDJET_PULSE_D ROUNDJET_RE ROUNDJET_RBF ROUNDJET_RBF_ITMAX ROUNDJET_RBF_TIMEOUT_SECONDS

[[ -f "$JULIA_PROJECT/Project.toml" ]] || {
    echo "Julia project not found: $JULIA_PROJECT/Project.toml" >&2
    echo "Set JULIA_PROJECT to the FLOWUnsteady environment directory." >&2
    exit 2
}

command -v "$JULIA" >/dev/null 2>&1 || {
    echo "Julia executable was not found: $JULIA" >&2
    exit 127
}

exec "$JULIA" --project="$JULIA_PROJECT" "$SCRIPT_DIR/training_sweep.jl" "$DATASET"
