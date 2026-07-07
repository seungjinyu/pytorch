#!/usr/bin/env bash
set -euo pipefail

CFG="$1"

RUN_ID=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['run_id'])")
RATIO=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['ratio'])")
CORES=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['cpu']['cores'])")
THREADS=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['cpu']['torch_threads'])")
MAX_STEPS=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['max_steps'])")
BATCH_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['batch_size'])")

EXP_DIR="experiments/results/${RUN_ID}"
mkdir -p "$EXP_DIR"

cp "$CFG" "$EXP_DIR/config.yaml"

export RUN_ID="$RUN_ID"
export EXP_DIR="$EXP_DIR"
export AUTO_DROP_RATIO="$RATIO"
export TORCH_THREADS="$THREADS"
export MAX_STEPS="$MAX_STEPS"
export BATCH_SIZE="$BATCH_SIZE"
export CSV_PATH="$EXP_DIR/app_node_a.csv"

echo "[RUN] $RUN_ID"
echo "[CPU] taskset -c $CORES threads=$THREADS"

OMP_NUM_THREADS="$THREADS" \
MKL_NUM_THREADS="$THREADS" \
OPENBLAS_NUM_THREADS="$THREADS" \
taskset -c "$CORES" \
python tests/test_node_a_resnet18.py \
  > "$EXP_DIR/stdout_a.log" 2>&1