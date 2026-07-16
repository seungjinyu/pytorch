#!/usr/bin/env bash

set -euo pipefail

RATIOS=(0.0 0.5 0.6 0.7 0.8 0.9)
REPEATS=20

PROFILE_CSV="./recompute_layer_profile.csv"
RESULT_CSV="./drop_ratio_experiment.csv"

NODE_B_LOG_DIR="./experiment_logs/node_b"
NODE_A_LOG_DIR="./experiment_logs/node_a"

mkdir -p "${NODE_B_LOG_DIR}"
mkdir -p "${NODE_A_LOG_DIR}"

if [[ ! -f "${PROFILE_CSV}" ]]; then
    echo "[ERROR] Missing profile CSV: ${PROFILE_CSV}"
    echo "Run: python3 tests/test_profile_resnet18.py"
    exit 1
fi

# 실험 시작할 때만 기존 통합 결과 삭제
rm -f "${RESULT_CSV}"

cleanup_node_b() {
    if [[ -n "${NODE_B_PID:-}" ]]; then
        kill "${NODE_B_PID}" 2>/dev/null || true
        wait "${NODE_B_PID}" 2>/dev/null || true
        unset NODE_B_PID
    fi
}

trap cleanup_node_b EXIT INT TERM

for ratio in "${RATIOS[@]}"; do
    for run_id in $(seq 1 "${REPEATS}"); do
        echo
        echo "========================================"
        echo "[EXPERIMENT] ratio=${ratio} run=${run_id}"
        echo "========================================"

        JIN_RECOMPUTE_PROFILE_PATH="${PROFILE_CSV}" \
        python3 -u tests/test_node_b_resnet18.py \
            > "${NODE_B_LOG_DIR}/ratio_${ratio}_run_${run_id}.log" 2>&1 &

        NODE_B_PID=$!

        sleep 8

        if ! kill -0 "${NODE_B_PID}" 2>/dev/null; then
            echo "[ERROR] Node B terminated early."
            tail -n 50 \
                "${NODE_B_LOG_DIR}/ratio_${ratio}_run_${run_id}.log"
            exit 1
        fi

        JIN_AUTO_DROP_RATIO="${ratio}" \
        JIN_EXPERIMENT_RUN_ID="${run_id}" \
        python3 -u tests/test_node_a_resnet18.py \
            > "${NODE_A_LOG_DIR}/ratio_${ratio}_run_${run_id}.log" 2>&1

        cleanup_node_b

        echo "[DONE] ratio=${ratio} run=${run_id}"
        tail -n 1 "${RESULT_CSV}"
    done
done

echo
echo "========================================"
echo "[ALL EXPERIMENTS COMPLETE]"
echo "Result: ${RESULT_CSV}"
echo "========================================"

cat "${RESULT_CSV}"
