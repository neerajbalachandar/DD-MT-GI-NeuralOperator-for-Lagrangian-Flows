#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="/home/dysco/FLOWUnsteady/Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward"
FLOWUNSTEADY_PROJECT="${FLOWUNSTEADY_PROJECT:-/home/dysco/FLOWUnsteady}"
PYTHON_BIN="${TASK1_PIPELINE_PYTHON:-python3}"
TASK1_DEVICE="${TASK1_DEVICE:-auto}"

cd "${WORKFLOW_DIR}"

echo "Task1 overnight pipeline"
echo "Workflow: ${WORKFLOW_DIR}"
echo "Julia project: ${FLOWUNSTEADY_PROJECT}"
echo "Python: ${PYTHON_BIN}"
echo "Task1 predictor device: ${TASK1_DEVICE}"

export FLOWUNSTEADY_PROJECT
export TASK1_PYTHON="${TASK1_PYTHON:-${PYTHON_BIN}}"

"${PYTHON_BIN}" benchmark_comparison_task1.py \
  --run \
  --project "${FLOWUNSTEADY_PROJECT}" \
  --device "${TASK1_DEVICE}" \
  --baseline-dir output/runtime_baseline_fmm \
  --surrogate-dir output/runtime_surrogate_ml \
  --out-dir output/benchmark_task1 \
  "$@"
