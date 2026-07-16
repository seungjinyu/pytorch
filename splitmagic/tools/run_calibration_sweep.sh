#!/bin/bash

set -euo pipefail

CAPS=(0.2 0.4 0.6 0.8 0.9 0.99)
REPEATS="${REPEATS:-3}"
COMPLETED_RUNS=0
ENDPOINT="${ENDPOINT:-tcp://10.32.137.57:5555}"
NETWORK_MBPS="${NETWORK_MBPS:-100}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CALIBRATION_OUTPUT="${CALIBRATION_OUTPUT:-./calibration_runs.csv}"
RECOMPUTE_MENU="${RECOMPUTE_MENU:?RECOMPUTE_MENU required}"

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "${PROJECT_ROOT}"

CALIBRATION_OUTPUT="$(realpath -m "${CALIBRATION_OUTPUT}")"
RECOMPUTE_MENU="$(realpath "${RECOMPUTE_MENU}")"

OUTPUT_DIR="$(dirname "${CALIBRATION_OUTPUT}")"
NODE_A_TIMING_CSV="${OUTPUT_DIR}/node_a_calibration.csv"

mkdir -p "${OUTPUT_DIR}"

RUN_ID=1

rm -f "${CALIBRATION_OUTPUT}"
rm -f "${NODE_A_TIMING_CSV}"

echo "[CALIBRATION] output=${CALIBRATION_OUTPUT}"
echo "[CALIBRATION] menu=${RECOMPUTE_MENU}"

for cap in "${CAPS[@]}"; do
    for repeat in $(seq 1 "${REPEATS}"); do
        NODE_A_LOG="/tmp/node_a_calibration_${RUN_ID}.log"

        echo
        echo "===================================="
        echo "cap=${cap}"
        echo "repeat=${repeat}/${REPEATS}"
        echo "run_id=${RUN_ID}"
        echo "log=${NODE_A_LOG}"
        echo "===================================="

        JIN_SELECTION_POLICY=cost \
        JIN_RECOMPUTE_COST_CSV="${RECOMPUTE_MENU}" \
        JIN_NETWORK_MBPS="${NETWORK_MBPS}" \
        JIN_ENDPOINT="${ENDPOINT}" \
        JIN_MIN_BENEFIT_MS=0.0 \
        JIN_MAX_COST_DROP_RATIO="${cap}" \
        JIN_AUTO_DROP_RATIO=0.0 \
        JIN_EXPERIMENT_RUN_ID="${RUN_ID}" \
        JIN_MAX_STEPS=1 \
        JIN_EXPERIMENT_CSV="${CALIBRATION_OUTPUT}" \
        JIN_NODE_A_CSV="${NODE_A_TIMING_CSV}" \
        "${PYTHON_BIN}" -u tests/test_node_a_cost_resnet18.py \
        2>&1 | tee "${NODE_A_LOG}"

        if [[ ! -f "${CALIBRATION_OUTPUT}" ]]; then
            echo "[ERROR] calibration CSV was not created"
            exit 1
        fi

        expected_lines=$((RUN_ID + 1))
        actual_lines="$(wc -l < "${CALIBRATION_OUTPUT}")"

        if [[ "${actual_lines}" -lt "${expected_lines}" ]]; then
            echo "[ERROR] calibration row was not written"
            echo "[ERROR] expected_lines>=${expected_lines}"
            echo "[ERROR] actual_lines=${actual_lines}"
            echo "[ERROR] log=${NODE_A_LOG}"
            tail -100 "${NODE_A_LOG}" || true
            exit 1
        fi

        echo \
            "[CALIBRATION_ROW_WRITTEN] " \
            "run_id=${RUN_ID} " \
            "cap=${cap} " \
            "lines=${actual_lines}"

        RUN_ID=$((RUN_ID + 1))
        sleep 2
    done
done

echo
echo "[CALIBRATION DONE]"
echo "[CSV] ${CALIBRATION_OUTPUT}"