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
PROFILE_REPEAT="${PROFILE_REPEAT:-100}"
PROFILE_WARMUP="${PROFILE_WARMUP:-10}"

OUTPUT_ROOT="${OUTPUT_ROOT:-profiles}"
PROFILE_ID="${MODEL}_bs${BATCH_SIZE}_${DEVICE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${PROFILE_ID}"

LAYER_PROFILE="${OUTPUT_DIR}/recompute_layer_profile.csv"
RECOMPUTE_MENU="${OUTPUT_DIR}/recompute_menu.csv"
METADATA_JSON="${OUTPUT_DIR}/node_b_metadata.json"

mkdir -p "${OUTPUT_DIR}"

echo "===================================="
echo "[NODE B COST MODEL PREPARE]"
echo "hostname=$(hostname)"
echo "model=${MODEL}"
echo "batch_size=${BATCH_SIZE}"
echo "device=${DEVICE}"
echo "output_dir=${OUTPUT_DIR}"
echo "===================================="

python3 -u tools/profile_recompute_layers.py \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --repeat "${PROFILE_REPEAT}" \
    --warmup "${PROFILE_WARMUP}" \
    --output "${LAYER_PROFILE}"

PROFILE_ROWS="$(
    tail -n +2 "${LAYER_PROFILE}" |
    wc -l
)"

if [[ "${PROFILE_ROWS}" -lt 69 ]]; then
    echo "[ERROR] incomplete layer profile"
    echo "[ERROR] expected at least 69 rows"
    echo "[ERROR] actual=${PROFILE_ROWS}"
    exit 1
fi

bash tools/build_recompute_menu.sh \
    "${LAYER_PROFILE}" \
    "${RECOMPUTE_MENU}"

python3 - <<PY
import json
import platform
import time
import torch

metadata = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "hostname": platform.node(),
    "model": "${MODEL}",
    "batch_size": ${BATCH_SIZE},
    "device": "${DEVICE}",
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "layer_profile": "${LAYER_PROFILE}",
    "recompute_menu": "${RECOMPUTE_MENU}",
}

with open("${METADATA_JSON}", "w") as f:
    json.dump(metadata, f, indent=2)

print(json.dumps(metadata, indent=2))
PY

echo
echo "[NODE B PREPARE DONE]"
echo "[LAYER PROFILE] ${LAYER_PROFILE}"
echo "[RECOMPUTE MENU] ${RECOMPUTE_MENU}"
echo "[METADATA] ${METADATA_JSON}"