#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL="${MODEL:-resnet18}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"

PROFILE_ID="${MODEL}_bs${BATCH_SIZE}_${DEVICE}"
OUTPUT_DIR="${PROJECT_ROOT}/profiles/${PROFILE_ID}"

LAYER_PROFILE="${OUTPUT_DIR}/recompute_layer_profile.csv"
RECOMPUTE_MENU="${OUTPUT_DIR}/recompute_menu.csv"
CALIBRATION_CSV="${OUTPUT_DIR}/calibration_runs.csv"
CALIBRATION_JSON="${OUTPUT_DIR}/recompute_calibration.json"
NODE_A_METADATA="${OUTPUT_DIR}/node_a_metadata.json"

NETWORK_MBPS="${NETWORK_MBPS:-100}"
ENDPOINT="${ENDPOINT:-tcp://10.32.137.57:5555}"
REPEATS="${REPEATS:-3}"

if [[ ! -f "${LAYER_PROFILE}" ]]; then
    echo "[ERROR] missing layer profile: ${LAYER_PROFILE}"
    exit 1
fi

if [[ ! -f "${RECOMPUTE_MENU}" ]]; then
    echo "[ERROR] missing recompute menu: ${RECOMPUTE_MENU}"
    exit 1
fi

echo "===================================="
echo "[NODE A CALIBRATION PREPARE]"
echo "hostname=$(hostname)"
echo "output_dir=${OUTPUT_DIR}"
echo "network_mbps=${NETWORK_MBPS}"
echo "endpoint=${ENDPOINT}"
echo "repeats=${REPEATS}"
echo "===================================="

CALIBRATION_OUTPUT="${CALIBRATION_CSV}" \
RECOMPUTE_MENU="${RECOMPUTE_MENU}" \
NETWORK_MBPS="${NETWORK_MBPS}" \
ENDPOINT="${ENDPOINT}" \
REPEATS="${REPEATS}" \
bash tools/run_calibration_sweep.sh

python3 -u tools/build_cost_calibration.py \
    --input "${CALIBRATION_CSV}" \
    --output "${CALIBRATION_JSON}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}"

python3 - <<PY
import json
import platform
import time

metadata = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "hostname": platform.node(),
    "model": "${MODEL}",
    "batch_size": ${BATCH_SIZE},
    "recompute_device": "${DEVICE}",
    "network_mbps": float("${NETWORK_MBPS}"),
    "endpoint": "${ENDPOINT}",
    "layer_profile": "${LAYER_PROFILE}",
    "recompute_menu": "${RECOMPUTE_MENU}",
    "calibration_csv": "${CALIBRATION_CSV}",
    "calibration_json": "${CALIBRATION_JSON}",
}

with open("${NODE_A_METADATA}", "w") as f:
    json.dump(metadata, f, indent=2)

print(json.dumps(metadata, indent=2))
PY

echo
echo "[NODE A CALIBRATION DONE]"
echo "[CALIBRATION CSV] ${CALIBRATION_CSV}"
echo "[CALIBRATION JSON] ${CALIBRATION_JSON}"