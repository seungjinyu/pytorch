#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# SplitMagic local A/B experiment with:
#   - Linux network namespaces
#   - veth virtual link
#   - bandwidth/latency emulation
#   - Nsight Systems profiling
#
# Usage:
#   sudo -E ./run_local_nsys.sh
#
# Optional:
#   NETWORK_MBPS=100 LATENCY_MS=10 RUN_ID=0 sudo -E ./run_local_nsys.sh
# ============================================================


# ------------------------------------------------------------
# 1. User configuration
# ------------------------------------------------------------

MODEL="${MODEL:-resnet18}"

PYTORCH_ROOT="${PYTORCH_ROOT:-/home/syu23/seungjin/pytorch}"
TORCHVISION_ROOT="${TORCHVISION_ROOT:-/home/syu23/torchvision-0.17}"
PROJECT_DIR="${PROJECT_DIR:-/home/syu23/seungjin/pytorch/splitmagic}"

PYTHONPATH_VALUE="${PYTORCH_ROOT}:${TORCHVISION_ROOT}:${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/syu23/miniconda3/envs/torch-build/bin/python}"
# 실제 실행 파일 이름에 맞게 수정
NODE_A_CMD=(
    "$PYTHON_BIN"
    "tests/test_node_a_${MODEL}.py"
)

NODE_B_CMD=(
    "$PYTHON_BIN"
    "tests/test_node_b_${MODEL}.py"
)

# NODE_A_CMD=(
#     "$PYTHON_BIN"
#     "tests/test_node_a_vgg.py"
# )

# NODE_B_CMD=(
#     "$PYTHON_BIN"
#     "tests/test_node_b_vgg.py"
# )

# A/B endpoint
PORT="${PORT:-5555}"
NODE_A_IP="${NODE_A_IP:-10.10.0.1}"
NODE_B_IP="${NODE_B_IP:-10.10.0.2}"
SUBNET_CIDR="${SUBNET_CIDR:-24}"

NODE_A_ENDPOINT="tcp://${NODE_B_IP}:${PORT}"
NODE_B_ENDPOINT="tcp://${NODE_B_IP}:${PORT}"

# Network emulation
NETWORK_MBPS="${NETWORK_MBPS:-1000}"
LATENCY_MS="${LATENCY_MS:-0}"
JITTER_MS="${JITTER_MS:-0}"
QUEUE_LIMIT="${QUEUE_LIMIT:-1000}"

# Experiment configuration
RUN_ID="${RUN_ID:-0}"
DROP_RATIO="${DROP_RATIO:-0.5}"
MPS_PERCENT="${MPS_PERCENT:-100}"
MAX_STEPS="${MAX_STEPS:-5}"

# Namespace names
NS_A="${NS_A:-splitmagic_a}"
NS_B="${NS_B:-splitmagic_b}"

VETH_A="${VETH_A:-veth_sm_a}"
VETH_B="${VETH_B:-veth_sm_b}"

# Output
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/nsys_results}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="${RUN_NAME:-${TIMESTAMP}_${MODEL}_${NETWORK_MBPS}mbps_${LATENCY_MS}ms_ratio${DROP_RATIO}_mps${MPS_PERCENT}_run${RUN_ID}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

TEMPLATE_PLAN_PATH="${RUN_DIR}/jin_template_plan.tsv"
TEMPLATE_PLAN_A_PATH="${RUN_DIR}/jin_template_plan_a.tsv"
EXECUTION_PLAN_PATH="${RUN_DIR}/jin_execution_plan.tsv"
ALIAS_PATH="${RUN_DIR}/jin_payload_recv.bin.alias"

NODE_A_LOG="${RUN_DIR}/node_a.log"
NODE_B_LOG="${RUN_DIR}/node_b.log"

# NSYS_OUTPUT="${RUN_DIR}/splitmagic_trace"
NSYS_OUTPUT="${RUN_DIR}/node_b_${NETWORK_MBPS}_${MPS_PERCENT}_${DROP_RATIO}"

# Timeouts
NODE_B_START_TIMEOUT="${NODE_B_START_TIMEOUT:-30}"
EXPERIMENT_TIMEOUT="${EXPERIMENT_TIMEOUT:-300}"


SELECTION_POLICY="${SELECTION_POLICY:-cost}"

RECOMPUTE_COST_CSV="${RECOMPUTE_COST_CSV:-${PROJECT_DIR}/menus_offline/${MODEL}_cuda_mps${MPS_PERCENT}/final_menu.csv}"

RECOMPUTE_LAYER_PROFILE_CSV="${RECOMPUTE_LAYER_PROFILE_CSV:-${PROJECT_DIR}/merged_profiles/${MODEL}_cuda_mps${MPS_PERCENT}_recompute_profile.csv}"

# echo "$CUDA_MPS_PIPE_DIRECTORY"
# echo "$CUDA_MPS_LOG_DIRECTORY"
# echo "$CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------
# 2. Internal state
# ------------------------------------------------------------

NODE_A_PID=""
NODE_B_PID=""

CREATED_NS_A=0
CREATED_NS_B=0


# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------

log() {
    printf '[RUN][%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf '[RUN][ERROR] %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 ||
        die "Required command not found: $1"
}

delete_namespaces() {
    if ip netns list | awk '{print $1}' | grep -qx "$NS_A"; then
        ip netns delete "$NS_A" || true
    fi

    if ip netns list | awk '{print $1}' | grep -qx "$NS_B"; then
        ip netns delete "$NS_B" || true
    fi
}

kill_process_group() {
    local pgid="${1:-}"

    [[ -n "$pgid" ]] || return 0

    if kill -0 -- "-$pgid" 2>/dev/null; then
        log "Stopping process group: $pgid"

        kill -TERM -- "-$pgid" 2>/dev/null || true

        for _ in {1..30}; do
            if ! kill -0 -- "-$pgid" 2>/dev/null; then
                return 0
            fi
            sleep 0.1
        done

        log "Force killing process group: $pgid"
        kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
}

kill_namespace_processes() {
    local ns="$1"

    if ip netns list | awk '{print $1}' | grep -qx "$ns"; then
        local pids
        pids="$(ip netns pids "$ns" 2>/dev/null || true)"

        if [[ -n "$pids" ]]; then
            kill -TERM $pids 2>/dev/null || true
            sleep 2
            kill -KILL $pids 2>/dev/null || true
        fi
    fi
}

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    log "Cleaning up..."

    # 먼저 nsys가 종료·flush할 기회를 줌
    kill_process_group "$NODE_A_PID"
    kill_process_group "$NODE_B_PID"

    # 남은 namespace 프로세스만 최후에 제거
    kill_namespace_processes "$NS_A"
    kill_namespace_processes "$NS_B"

    delete_namespaces

    log "Cleanup complete"

    exit "$exit_code"
}
on_interrupt() {
    trap - INT TERM
    log "Ctrl+C received."
    exit 130
}

on_terminate() {
    trap - INT TERM
    log "TERM received."
    exit 143
}

trap cleanup EXIT
trap on_interrupt INT
trap on_terminate TERM


# ------------------------------------------------------------
# 4. Validation
# ------------------------------------------------------------

[[ $EUID -eq 0 ]] ||
    die "Run this script with sudo -E."

require_command ip
require_command tc
require_command nsys
require_command timeout
require_command ss
require_command tee

[[ -d "$PROJECT_DIR" ]] ||
    die "PROJECT_DIR does not exist: $PROJECT_DIR"

if [[ "$SELECTION_POLICY" == "cost" ]]; then
    [[ -f "$RECOMPUTE_COST_CSV" ]] ||
        die "Recompute cost menu not found: $RECOMPUTE_COST_CSV"

    [[ -f "$RECOMPUTE_LAYER_PROFILE_CSV" ]] ||
        die "Recompute layer profile not found: $RECOMPUTE_LAYER_PROFILE_CSV"
fi

log "Selection policy  : ${SELECTION_POLICY}"
log "Cost menu         : ${RECOMPUTE_COST_CSV}"
log "Layer profile     : ${RECOMPUTE_LAYER_PROFILE_CSV}"

mkdir -p "$RUN_DIR"

chown -R "${SUDO_USER:-syu23}:${SUDO_USER:-syu23}" "$RUN_DIR"

log "Project directory : $PROJECT_DIR"
log "Output directory  : $RUN_DIR"
log "Python binary     : $PYTHON_BIN"
log "PYTHONPATH        : $PYTHONPATH_VALUE"
log "Network           : ${NETWORK_MBPS} Mbps"
log "Latency           : ${LATENCY_MS} ms"
log "Drop ratio        : ${DROP_RATIO}"
log "MPS percent       : ${MPS_PERCENT}%"
log "Run ID            : ${RUN_ID}"

[[ -x "$PYTHON_BIN" ]] ||
    die "Python binary not executable: $PYTHON_BIN"

if ! env \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    "$PYTHON_BIN" -c '
import sys
import torch
import torchvision

print("[RUN][PYTHON]", sys.executable)
print("[RUN][TORCH]", torch.__version__, torch.__file__)
print("[RUN][TORCHVISION]", torchvision.__version__, torchvision.__file__)
'; then
    die "Python, torch, or torchvision environment validation failed."
fi

log "Removing stale JIN temporary files..."

rm -f \
    /tmp/jin_execution_plan.tsv \
    /tmp/jin_template_plan.tsv \
    /tmp/jin_execution_plan_a.tsv \
    /tmp/jin_execution_plan_b.tsv

# ------------------------------------------------------------
# 5. Recreate namespaces
# ------------------------------------------------------------

log "Creating network namespaces..."

delete_namespaces

ip netns add "$NS_A"
CREATED_NS_A=1

ip netns add "$NS_B"
CREATED_NS_B=1

ip link add "$VETH_A" type veth peer name "$VETH_B"

ip link set "$VETH_A" netns "$NS_A"
ip link set "$VETH_B" netns "$NS_B"

ip -n "$NS_A" link set lo up
ip -n "$NS_B" link set lo up

ip -n "$NS_A" addr add \
    "${NODE_A_IP}/${SUBNET_CIDR}" \
    dev "$VETH_A"

ip -n "$NS_B" addr add \
    "${NODE_B_IP}/${SUBNET_CIDR}" \
    dev "$VETH_B"

ip -n "$NS_A" link set "$VETH_A" up
ip -n "$NS_B" link set "$VETH_B" up


# ------------------------------------------------------------
# 6. Apply network shaping
# ------------------------------------------------------------

log "Applying traffic control..."

# A -> B direction:
# qdisc on A's egress limits payload transmission toward B.
if [[ "$LATENCY_MS" != "0" || "$JITTER_MS" != "0" ]]; then
    ip netns exec "$NS_A" tc qdisc replace \
        dev "$VETH_A" root handle 1: \
        netem \
        delay "${LATENCY_MS}ms" "${JITTER_MS}ms" \
        rate "${NETWORK_MBPS}mbit" \
        limit "$QUEUE_LIMIT"
else
    ip netns exec "$NS_A" tc qdisc replace \
        dev "$VETH_A" root handle 1: \
        tbf \
        rate "${NETWORK_MBPS}mbit" \
        burst 1mb \
        latency 100ms
fi

# B -> A direction:
# This also limits the reply, including updated_state_dict.
if [[ "$LATENCY_MS" != "0" || "$JITTER_MS" != "0" ]]; then
    ip netns exec "$NS_B" tc qdisc replace \
        dev "$VETH_B" root handle 1: \
        netem \
        delay "${LATENCY_MS}ms" "${JITTER_MS}ms" \
        rate "${NETWORK_MBPS}mbit" \
        limit "$QUEUE_LIMIT"
else
    ip netns exec "$NS_B" tc qdisc replace \
        dev "$VETH_B" root handle 1: \
        tbf \
        rate "${NETWORK_MBPS}mbit" \
        burst 1mb \
        latency 100ms
fi

log "Checking virtual network..."

ip netns exec "$NS_A" ping \
    -c 2 \
    -W 2 \
    "$NODE_B_IP" \
    >/dev/null

log "Virtual network is reachable"


# ------------------------------------------------------------
# 7. Save experiment metadata
# ------------------------------------------------------------

cat >"${RUN_DIR}/experiment.env" <<EOF
PROJECT_DIR=${PROJECT_DIR}
RUN_NAME=${RUN_NAME}
RUN_ID=${RUN_ID}
NETWORK_MBPS=${NETWORK_MBPS}
LATENCY_MS=${LATENCY_MS}
JITTER_MS=${JITTER_MS}
DROP_RATIO=${DROP_RATIO}
MPS_PERCENT=${MPS_PERCENT}
MAX_STEPS=${MAX_STEPS}
NODE_A_IP=${NODE_A_IP}
NODE_B_IP=${NODE_B_IP}
PORT=${PORT}
NODE_A_ENDPOINT=${NODE_A_ENDPOINT}
NODE_B_ENDPOINT=${NODE_B_ENDPOINT}
EOF

ip netns exec "$NS_A" tc -s qdisc show dev "$VETH_A" \
    >"${RUN_DIR}/tc_a_before.txt"

ip netns exec "$NS_B" tc -s qdisc show dev "$VETH_B" \
    >"${RUN_DIR}/tc_b_before.txt"


# ------------------------------------------------------------
# 8. Start Node B under Nsight Systems
# ------------------------------------------------------------

log "Starting Node B..."

RUN_USER="${SUDO_USER:-syu23}"

setsid \
ip netns exec "$NS_B" \
    sudo -u "$RUN_USER" --preserve-env \
    env \
        PATH="/usr/lib/nsight-systems/host-linux-x64:/home/syu23/miniconda3/envs/torch-build/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
        LD_LIBRARY_PATH="/usr/lib/nsight-systems/host-linux-x64:${LD_LIBRARY_PATH:-}" \
        PYTHONPATH="$PYTHONPATH_VALUE" \
        JIN_RUN_ID="$RUN_ID" \
        JIN_EXPERIMENT_RUN_ID="$RUN_ID" \
        JIN_NETWORK_MBPS="$NETWORK_MBPS" \
        JIN_AUTO_DROP_RATIO="$DROP_RATIO" \
        JIN_MAX_STEPS="$MAX_STEPS" \
        JIN_ENDPOINT="$NODE_B_ENDPOINT" \
        JIN_TEMPLATE_PLAN_PATH="$TEMPLATE_PLAN_PATH" \
        JIN_EXECUTION_PLAN_PATH="$EXECUTION_PLAN_PATH" \
        JIN_ALIAS_PATH="$ALIAS_PATH" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}" \
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$MPS_PERCENT" \
    bash -c "
        cd '$PROJECT_DIR'

        exec nsys profile \
            --trace=cuda,nvtx,osrt \
            --sample=none \
            --force-overwrite=true \
            --output='$NSYS_OUTPUT' \
            \"\${@}\"
    " bash "${NODE_B_CMD[@]}" \
    > "$NODE_B_LOG" 2>&1 &

NODE_B_PID=$!

log "Node B PID/PGID: $NODE_B_PID"


# ------------------------------------------------------------
# 9. Wait until Node B listens
# ------------------------------------------------------------

log "Waiting for Node B to listen on ${NODE_B_IP}:${PORT}..."

server_ready=0

for ((i = 0; i < NODE_B_START_TIMEOUT; i++)); do
    if ! kill -0 "$NODE_B_PID" 2>/dev/null; then
        die "Node B exited before becoming ready. Check ${NODE_B_LOG}"
    fi

    if ip netns exec "$NS_B" ss -lnt |
        grep -q "${NODE_B_IP}:${PORT}"; then
        server_ready=1
        break
    fi

    sleep 1
done

[[ "$server_ready" -eq 1 ]] ||
    die "Node B did not listen within ${NODE_B_START_TIMEOUT}s."

log "Node B is ready"


# ------------------------------------------------------------
# 10. Run Node A
# ------------------------------------------------------------

log "Starting Node A..."

setsid \
timeout \
    --signal=TERM \
    --kill-after=10s \
    "${EXPERIMENT_TIMEOUT}s" \
    ip netns exec "$NS_A" \
    sudo -u "$RUN_USER" --preserve-env \
    env \
        PATH="/usr/lib/nsight-systems/host-linux-x64:/home/syu23/miniconda3/envs/torch-build/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
        LD_LIBRARY_PATH="/usr/lib/nsight-systems/host-linux-x64:${LD_LIBRARY_PATH:-}" \
        PYTHONPATH="$PYTHONPATH_VALUE" \
        JIN_RUN_ID="$RUN_ID" \
        JIN_EXPERIMENT_RUN_ID="$RUN_ID" \
        JIN_NETWORK_MBPS="$NETWORK_MBPS" \
        JIN_AUTO_DROP_RATIO="$DROP_RATIO" \
        JIN_MPS_PERCENT="$MPS_PERCENT" \
        JIN_SELECTION_POLICY="$SELECTION_POLICY" \
        JIN_RECOMPUTE_COST_CSV="$RECOMPUTE_COST_CSV" \
        JIN_TEMPLATE_PLAN_A_PATH="$TEMPLATE_PLAN_A_PATH" \
        JIN_MAX_STEPS="$MAX_STEPS" \
        JIN_ENDPOINT="$NODE_A_ENDPOINT" \
        SPLITMAGIC_RECOMPUTE_PROFILE_CSV="$RECOMPUTE_LAYER_PROFILE_CSV" \
    bash -c "
        cd '$PROJECT_DIR'

        exec nsys profile \
            --trace=nvtx,osrt \
            --sample=none \
            --force-overwrite=true \
            --output='${RUN_DIR}/node_a_${NETWORK_MBPS}_${MPS_PERCENT}_${DROP_RATIO}' \
            \"\${@}\"
    " bash "${NODE_A_CMD[@]}" \
    > "$NODE_A_LOG" 2>&1 &

NODE_A_PID=$!

log "Node A PID/PGID: $NODE_A_PID"

set +e
wait "$NODE_A_PID"
NODE_A_EXIT=$?
set -e

if [[ "$NODE_A_EXIT" -eq 124 ]]; then
    die "Node A timed out after ${EXPERIMENT_TIMEOUT}s."
elif [[ "$NODE_A_EXIT" -eq 130 ]]; then
    exit 130
elif [[ "$NODE_A_EXIT" -eq 143 ]]; then
    exit 143
elif [[ "$NODE_A_EXIT" -ne 0 ]]; then
    die "Node A failed with exit code ${NODE_A_EXIT}. Check ${NODE_A_LOG}"
fi

NODE_A_PID=""

log "Node A finished"


# ------------------------------------------------------------
# 11. Wait for Node B
# ------------------------------------------------------------

log "Waiting for Node B to finish..."

set +e

timeout \
    --signal=TERM \
    --kill-after=10s \
    60s \
    bash -c '
        pid="$1"

        while kill -0 "$pid" 2>/dev/null; do
            sleep 1
        done
    ' bash "$NODE_B_PID"

NODE_B_WAIT_EXIT=$?

set -e

if [[ "$NODE_B_WAIT_EXIT" -eq 124 ]]; then
    log "Node B did not stop automatically; terminating its process group."
    kill_process_group "$NODE_B_PID"
fi

wait "$NODE_B_PID" 2>/dev/null || true
NODE_B_PID=""


# ------------------------------------------------------------
# 12. Collect final traffic-control statistics
# ------------------------------------------------------------

ip netns exec "$NS_A" tc -s qdisc show dev "$VETH_A" \
    >"${RUN_DIR}/tc_a_after.txt" || true

ip netns exec "$NS_B" tc -s qdisc show dev "$VETH_B" \
    >"${RUN_DIR}/tc_b_after.txt" || true


# ------------------------------------------------------------
# 13. Validate Nsight output
# ------------------------------------------------------------

if [[ -f "${NSYS_OUTPUT}.nsys-rep" ]]; then
    NSYS_REPORT="${NSYS_OUTPUT}.nsys-rep"
elif [[ -f "${NSYS_OUTPUT}.qdrep" ]]; then
    NSYS_REPORT="${NSYS_OUTPUT}.qdrep"
else
    NSYS_REPORT=""
fi

if [[ -n "$NSYS_REPORT" ]]; then
    log "Nsight report created: $NSYS_REPORT"

    nsys stats \
        --report nvtxsum,cudaapisum,gpukernsum \
        "$NSYS_REPORT" \
        >"${RUN_DIR}/nsys_stats.txt" \
        2>"${RUN_DIR}/nsys_stats_error.txt" ||
        log "nsys stats could not generate every requested report."
else
    log "WARNING: Nsight report was not found."
fi


# ------------------------------------------------------------
# 14. Summary
# ------------------------------------------------------------

log "Experiment complete"
log "Node A log : $NODE_A_LOG"
log "Node B log : $NODE_B_LOG"

if [[ -n "$NSYS_REPORT" ]]; then
    log "NSYS report: $NSYS_REPORT"
fi

log "Results directory: $RUN_DIR"

# Prevent cleanup from changing successful exit status.
exit 0