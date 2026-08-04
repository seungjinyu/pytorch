#!/bin/bash

set -euo pipefail

LAYER_PROFILE="${1:?layer profile required}"
OUTPUT_MENU="${2:?output menu required}"

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

PYTHON_BIN="${PYTHON_BIN:-/home/syu23/miniconda3/envs/torch-build/bin/python3}"
NODE_B_LOG="${NODE_B_LOG:-/tmp/node_b_menu.log}"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 상대경로를 절대경로로 변환
LAYER_PROFILE="$(
    realpath "${LAYER_PROFILE}"
)"

OUTPUT_MENU_DIR="$(
    dirname "${OUTPUT_MENU}"
)"

mkdir -p "${OUTPUT_MENU_DIR}"

OUTPUT_MENU_DIR="$(
    realpath "${OUTPUT_MENU_DIR}"
)"

OUTPUT_MENU="${OUTPUT_MENU_DIR}/$(basename "${OUTPUT_MENU}")"

stop_node_b() {
    pkill -f '[t]ests/test_node_b_vgg.py' || true
    sleep 2
}

cleanup() {
    stop_node_b
}

trap cleanup EXIT

stop_node_b

echo "[MENU] layer_profile=${LAYER_PROFILE}"
echo "[MENU] output_menu=${OUTPUT_MENU}"
echo "[MENU] starting Node B"

rm -f "${NODE_B_LOG}"

nohup env \
    PYTHONPATH="${PYTHONPATH}" \
    JIN_BUILD_RECOMPUTE_MENU=1 \
    JIN_RECOMPUTE_PROFILE_PATH="${LAYER_PROFILE}" \
    JIN_RECOMPUTE_MENU_PATH="${OUTPUT_MENU}" \
    "${PYTHON_BIN}" -u \
    tests/test_node_b_vgg.py \
    > "${NODE_B_LOG}" 2>&1 \
    < /dev/null &

echo "[MENU] waiting for Node B"

ready=0

for _ in $(seq 1 60); do
    if grep -q '\[Node B\] listening' "${NODE_B_LOG}" 2>/dev/null
    then
        ready=1
        break
    fi

    if ! pgrep -f '[t]ests/test_node_b_vgg.py' >/dev/null
    then
        echo "[ERROR] Node B exited"
        tail -100 "${NODE_B_LOG}" || true
        exit 1
    fi

    sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
    echo "[ERROR] Node B did not become ready"
    tail -100 "${NODE_B_LOG}" || true
    exit 1
fi

echo "[MENU] Node B ready"

JIN_SELECTION_POLICY=none \
JIN_AUTO_DROP_RATIO=0.0 \
JIN_MAX_STEPS=1 \
JIN_ENDPOINT="tcp://127.0.0.1:5556" \
"${PYTHON_BIN}" -u \
tests/test_node_a_vgg.py

if [[ ! -f "${OUTPUT_MENU}" ]]; then
    echo "[ERROR] recompute menu was not created"
    tail -200 "${NODE_B_LOG}" || true
    exit 1
fi

echo "[MENU] created: ${OUTPUT_MENU}"

EXPECTED_HEADER="key,tensor_mb,start,target,path_len,recompute_ms,missing_profile,path"

ACTUAL_HEADER="$(
    head -1 "${OUTPUT_MENU}" |
    tr -d '\r'
)"

if [[ "${ACTUAL_HEADER}" != "${EXPECTED_HEADER}" ]]; then
    echo "[ERROR] unexpected recompute menu header"
    echo "[ERROR] expected=${EXPECTED_HEADER}"
    echo "[ERROR] actual=${ACTUAL_HEADER}"
    exit 1
fi