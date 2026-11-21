#!/usr/bin/env bash
# End-to-end VortexGINO pipeline: raw FLOWVPM data -> lossless archives -> tests -> training ->
# evaluation (metrics + plots).
#
#   bash scripts/run_full_pipeline.sh                      # full run (several hours)
#   RESUME=1 bash scripts/run_full_pipeline.sh             # continue an interrupted training run
#
# Environment variables (all optional):
#   PYTHON        python interpreter with the requirements installed      (default: python)
#   RAW_ROOT      raw FLOWVPM output (train/validate/test/vortexrings/...) (default: datagen_code/training_data)
#   DATA_ROOT     processed archives                                       (default: processed_data/vortex_rings_v2)
#   CONFIG        training config                                          (default: configs/vortex_gino_full.yaml)
#   RUN_DIR       output directory for checkpoints/logs/evaluation         (default: runs/vortex_gino_full)
#   RESUME=1      resume from $RUN_DIR/last_model.pt if it exists
#   SKIP_TESTS=1  skip the unit tests
#   TRAIN_ARGS    extra arguments for scripts/train_vortex.py (e.g. "--set training.epochs=2")
#   EVAL_ARGS     extra arguments for scripts/evaluate_vortex.py (e.g. "--quick")
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RAW_ROOT="${RAW_ROOT:-datagen_code/training_data}"
DATA_ROOT="${DATA_ROOT:-processed_data/vortex_rings_v2}"
CONFIG="${CONFIG:-configs/vortex_gino_full.yaml}"
RUN_DIR="${RUN_DIR:-runs/vortex_gino_full}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # deterministic cuBLAS (checkpointed rollouts recompute exactly)
export PYTHONUNBUFFERED=1

echo "== [0/4] environment"
"$PYTHON" - <<'EOF'
import torch, neuralop
print(f"torch {torch.__version__}, neuralop {neuralop.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB, compute capability {p.major}.{p.minor}")
    x = torch.randn(64, 64, device="cuda"); (x @ x).sum().item()   # fails early if the CUDA build lacks this GPU
else:
    raise SystemExit("No CUDA GPU visible: install a CUDA build of torch that supports your GPU.")
EOF

echo "== [1/4] build lossless archives from $RAW_ROOT -> $DATA_ROOT (skips cases already exported)"
"$PYTHON" scripts/build_vortex_dataset.py --raw-root "$RAW_ROOT" --out "$DATA_ROOT"

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    echo "== [2/4] unit tests (CPU)"
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -m pytest -q -p no:cacheprovider tests/test_vortex_operator.py tests/test_vortex_data.py
fi

echo "== [3/4] train ($CONFIG -> $RUN_DIR)"
mkdir -p "$RUN_DIR"
resume_args=()
if [[ "${RESUME:-0}" == "1" && -f "$RUN_DIR/last_model.pt" ]]; then
    resume_args=(--resume "$RUN_DIR/last_model.pt")
    echo "   resuming from $RUN_DIR/last_model.pt"
fi
# shellcheck disable=SC2086
"$PYTHON" -W ignore scripts/train_vortex.py --config "$CONFIG" \
    --set "data.root=$DATA_ROOT" --set "training.run_dir=$RUN_DIR" "${resume_args[@]}" ${TRAIN_ARGS:-} \
    2>&1 | tee -a "$RUN_DIR/train.log"

echo "== [4/4] evaluate $RUN_DIR/best_model.pt (val + test cases, 180-step rollouts, plots)"
# shellcheck disable=SC2086
"$PYTHON" -W ignore scripts/evaluate_vortex.py --checkpoint "$RUN_DIR/best_model.pt" \
    --data-root "$DATA_ROOT" --out "$RUN_DIR/evaluation" ${EVAL_ARGS:-} \
    2>&1 | tee "$RUN_DIR/evaluation.log"

echo
echo "Done."
echo "  summary : $RUN_DIR/evaluation/summary.md  (plain-language metric definitions included)"
echo "  per case: $RUN_DIR/evaluation/summary.csv, $RUN_DIR/evaluation/metrics.json"
echo "  plots   : $RUN_DIR/evaluation/plots/"
ls -1 "$RUN_DIR/evaluation/plots" | sed 's/^/            /'
