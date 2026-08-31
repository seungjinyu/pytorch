#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# SplitMagic remote A/B experiment
#
# Node A : Remote CPU machine (e.g., Raspberry Pi 5)
# Node B : Local GPU machine
#
# Features:
#   - SSH-based remote Node A execution
#   - Real network connection between Node A and Node B
#   - Optional tc bandwidth/latency shaping on both machines
#   - Nsight Systems profiling on local Node B
#   - MODEL=resnet18 / resnet18_imagenet
#
# Usage example:
#
#   sudo -E \
#       REMOTE_HOST=192.168.1.50 \
#       REMOTE_USER=syu23 \
#       NODE_B_IP=192.168.1.40 \
#       LOCAL_IFACE=enp3s0 \
#       REMOTE_IFACE=eth0 \
#       MODEL=resnet18 \
#       ./run_remote_nsys.sh
#
# NOTE:
#   tc is applied to the ROOT qdisc of LOCAL_IFACE and
#   REMOTE_IFACE. Prefer a dedicated Ethernet interface.
# ============================================================


# ------------------------------------------------------------
# 1. Model configuration
# ------------------------------------------------------------

MODEL="${MODEL:-resnet18}"

case "$MODEL" in

    resnet18)
        DATASET="cifar10"
        NUM_CLASSES=10
        NODE_A_SCRIPT="tests/test_node_a_resnet18.py"
        NODE_B_SCRIPT="tests/test_node_b_resnet18.py"
        ;;

    resnet18_imagenet)
        DATASET="imagenet"
        NUM_CLASSES=200
        NODE_A_SCRIPT="tests/test_node_a_resnet18_imagenet.py"
        NODE_B_SCRIPT="tests/test_node_b_resnet18_imagenet.py"
        ;;

    *)
        echo "[RUN][ERROR] Unknown MODEL: $MODEL" >&2
        exit 1
        ;;

esac


# ------------------------------------------------------------
# 2. Local GPU machine configuration
# ------------------------------------------------------------

PYTORCH_ROOT="${PYTORCH_ROOT:-/home/syu23/seungjin/pytorch}"
TORCHVISION_ROOT="${TORCHVISION_ROOT:-/home/syu23/torchvision-0.17}"
PROJECT_DIR="${PROJECT_DIR:-/home/syu23/seungjin/pytorch/splitmagic}"

PYTHON_BIN="${PYTHON_BIN:-/home/syu23/miniconda3/envs/torch-build/bin/python}"

PYTHONPATH_VALUE="${PYTORCH_ROOT}:${TORCHVISION_ROOT}:${PROJECT_DIR}"

NODE_B_CMD=(
    "$PYTHON_BIN"
    "$NODE_B_SCRIPT"
)


# ------------------------------------------------------------
# 3. Remote Node A configuration
# ------------------------------------------------------------

REMOTE_HOST="${REMOTE_HOST:-10.32.126.244}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"

REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/ubuntu/splitmagic_project}"

REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-/home/ubuntu/venv/splitmagic/bin/python}"

REMOTE_PYTHONPATH_VALUE="${REMOTE_PROJECT_DIR}"

REMOTE_NODE_A_SCRIPT="${REMOTE_PROJECT_DIR}/${NODE_A_SCRIPT}"

# CPU threads used by Node A on Raspberry Pi
NODE_A_THREADS="${NODE_A_THREADS:-4}"

# CPU-side PyTorch threads used by Node B
NODE_B_THREADS="${NODE_B_THREADS:-1}"

# Normally keep this OFF on Raspberry Pi.
# Set REMOTE_USE_NSYS=1 only if nsys is installed and works there.
REMOTE_USE_NSYS="${REMOTE_USE_NSYS:-1}"


# ------------------------------------------------------------
# 4. SSH configuration
# ------------------------------------------------------------

SSH_PORT="${SSH_PORT:-22}"

RUN_USER="${SUDO_USER:-syu23}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

SSH_OPTS=(
    -p "$SSH_PORT"
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=4
    -o UserKnownHostsFile="${RUN_HOME}/.ssh/known_hosts"
)

REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

run_ssh() {
    sudo -H -u "$RUN_USER" \
        ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "$@"
}

run_scp() {
    sudo -H -u "$RUN_USER" \
        scp -P "$SSH_PORT" "$@"
}


# ------------------------------------------------------------
# 5. A/B endpoint
# ------------------------------------------------------------

PORT="${PORT:-5555}"

# This MUST be an IP address reachable from Raspberry Pi.
#
# Example:
#   NODE_B_IP=192.168.1.40
#
NODE_B_IP="${NODE_B_IP:-}"

NODE_A_ENDPOINT="tcp://${NODE_B_IP}:${PORT}"
NODE_B_ENDPOINT="tcp://${NODE_B_IP}:${PORT}"


# ------------------------------------------------------------
# 6. Network shaping
# ------------------------------------------------------------

NETWORK_MBPS="${NETWORK_MBPS:-1000}"
LATENCY_MS="${LATENCY_MS:-0}"
JITTER_MS="${JITTER_MS:-0}"
QUEUE_LIMIT="${QUEUE_LIMIT:-1000}"

# 1 = use tc
# 0 = real network without tc shaping
ENABLE_TC="${ENABLE_TC:-1}"

# Local GPU node interface
# Example:
#   enp3s0
LOCAL_IFACE="${LOCAL_IFACE:-}"

# Raspberry Pi interface
# Usually:
#   eth0
REMOTE_IFACE="${REMOTE_IFACE:-eth0}"


# ------------------------------------------------------------
# 7. Experiment configuration
# ------------------------------------------------------------

RUN_ID="${RUN_ID:-0}"
DROP_RATIO="${DROP_RATIO:-0.5}"
MPS_PERCENT="${MPS_PERCENT:-100}"
MAX_STEPS="${MAX_STEPS:-5}"
BATCH_SIZE="${JIN_BATCH_SIZE:-32}"
SELECTION_POLICY="${SELECTION_POLICY:-ratio}"


# ------------------------------------------------------------
# 8. Output
# ------------------------------------------------------------

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/nsys_results_remote}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="${RUN_NAME:-${TIMESTAMP}_${MODEL}_bs${BATCH_SIZE}_${NETWORK_MBPS}mbps_${LATENCY_MS}ms_policy_${SELECTION_POLICY}_maxratio${DROP_RATIO}_mps${MPS_PERCENT}_run${RUN_ID}}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

TEMPLATE_PLAN_PATH="${RUN_DIR}/jin_template_plan.tsv"
TEMPLATE_PLAN_A_PATH="${REMOTE_PROJECT_DIR}/.splitmagic_remote/jin_template_plan_a.tsv"
EXECUTION_PLAN_PATH="${RUN_DIR}/jin_execution_plan.tsv"
ALIAS_PATH="${RUN_DIR}/jin_payload_recv.bin.alias"

NODE_A_LOG="${RUN_DIR}/node_a.log"
NODE_B_LOG="${RUN_DIR}/node_b.log"

NSYS_OUTPUT="${RUN_DIR}/node_b_${NETWORK_MBPS}_${MPS_PERCENT}_${DROP_RATIO}"


# ------------------------------------------------------------
# 9. Timeouts
# ------------------------------------------------------------

NODE_B_START_TIMEOUT="${NODE_B_START_TIMEOUT:-60}"
EXPERIMENT_TIMEOUT="${EXPERIMENT_TIMEOUT:-1200}"
NODE_B_FINISH_TIMEOUT="${NODE_B_FINISH_TIMEOUT:-120}"


# ------------------------------------------------------------
# 10. Recompute profiles
# ------------------------------------------------------------

RECOMPUTE_COST_CSV="${RECOMPUTE_COST_CSV:-${PROJECT_DIR}/menus_offline/${MODEL}_cuda_mps${MPS_PERCENT}/final_menu.csv}"

RECOMPUTE_LAYER_PROFILE_CSV="${RECOMPUTE_LAYER_PROFILE_CSV:-${PROJECT_DIR}/merged_profiles/${MODEL}_cuda_mps${MPS_PERCENT}_recompute_profile.csv}"

# Remote equivalents.
# Only needed when selection_policy=cost.
REMOTE_RECOMPUTE_COST_CSV="${REMOTE_RECOMPUTE_COST_CSV:-${REMOTE_PROJECT_DIR}/menus_offline/${MODEL}_cuda_mps${MPS_PERCENT}/final_menu.csv}"

REMOTE_RECOMPUTE_LAYER_PROFILE_CSV="${REMOTE_RECOMPUTE_LAYER_PROFILE_CSV:-${REMOTE_PROJECT_DIR}/merged_profiles/${MODEL}_cuda_mps${MPS_PERCENT}_recompute_profile.csv}"


# ------------------------------------------------------------
# 11. Internal state
# ------------------------------------------------------------

NODE_A_PID=""
NODE_B_PID=""

LOCAL_TC_APPLIED=0
REMOTE_TC_APPLIED=0


# ------------------------------------------------------------
# 12. Helpers
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


remote_exec() {

    ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" "$@"

}


remove_local_tc() {

    [[ "$LOCAL_TC_APPLIED" -eq 1 ]] || return 0

    log "Removing local tc qdisc from ${LOCAL_IFACE}"

    tc qdisc del \
        dev "$LOCAL_IFACE" \
        root \
        2>/dev/null || true

    LOCAL_TC_APPLIED=0
}


remove_remote_tc() {

    [[ "$REMOTE_TC_APPLIED" -eq 1 ]] || return 0

    log "Removing remote tc qdisc from ${REMOTE_IFACE}"

    run_ssh \
        "sudo tc qdisc del dev '$REMOTE_IFACE' root 2>/dev/null || true" \
        || true

    REMOTE_TC_APPLIED=0
}


kill_remote_node_a() {

    [[ -n "$REMOTE_HOST" ]] || return 0

    log "Stopping possible stale remote Node A..."

    run_ssh \
        "pkill -TERM -f '${NODE_A_SCRIPT}' 2>/dev/null || true" \
        >/dev/null 2>&1 || true
}


cleanup() {

    local exit_code=$?

    trap - EXIT INT TERM

    log "Cleaning up..."

    kill_process_group "$NODE_A_PID"
    kill_process_group "$NODE_B_PID"

    kill_remote_node_a

    remove_local_tc
    remove_remote_tc

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
# 13. Validation
# ------------------------------------------------------------

[[ $EUID -eq 0 ]] ||
    die "Run this script with sudo -E."

require_command ssh
require_command tc
require_command nsys
require_command timeout
require_command ss
require_command tee

[[ -n "$REMOTE_HOST" ]] ||
    die "REMOTE_HOST is required."

[[ -n "$NODE_B_IP" ]] ||
    die "NODE_B_IP is required."

if [[ "$ENABLE_TC" == "1" ]]; then

    [[ -n "$LOCAL_IFACE" ]] ||
        die "LOCAL_IFACE is required when ENABLE_TC=1."

fi

[[ -d "$PROJECT_DIR" ]] ||
    die "PROJECT_DIR does not exist: $PROJECT_DIR"

[[ -x "$PYTHON_BIN" ]] ||
    die "Python binary not executable: $PYTHON_BIN"


# ------------------------------------------------------------
# 14. Local Python validation
# ------------------------------------------------------------

log "Validating local Python environment..."

env \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    "$PYTHON_BIN" - <<'PY'
import sys
import torch
import torchvision

print("[RUN][LOCAL][PYTHON]", sys.executable)
print("[RUN][LOCAL][TORCH]", torch.__version__, torch.__file__)
print("[RUN][LOCAL][TORCHVISION]", torchvision.__version__, torchvision.__file__)
PY


# ------------------------------------------------------------
# 15. SSH validation
# ------------------------------------------------------------

log "Checking SSH connection to ${REMOTE_TARGET}..."

run_ssh \
    "echo '[RUN][REMOTE] SSH connection OK'" \
    || die "Cannot connect to ${REMOTE_TARGET}"


# ------------------------------------------------------------
# 16. Remote environment validation
# ------------------------------------------------------------

log "Validating remote Python environment..."

run_ssh \
    "env \
        PYTHONPATH='$REMOTE_PYTHONPATH_VALUE' \
        '$REMOTE_PYTHON_BIN' -c \"
import sys
import torch
import torchvision

print('[RUN][REMOTE][PYTHON]', sys.executable)
print('[RUN][REMOTE][TORCH]', torch.__version__, torch.__file__)
print('[RUN][REMOTE][TORCHVISION]', torchvision.__version__, torchvision.__file__)
\"" \
    || die "Remote Python environment validation failed."


# ------------------------------------------------------------
# 17. Validate model scripts
# ------------------------------------------------------------

[[ -f "${PROJECT_DIR}/${NODE_B_SCRIPT}" ]] ||
    die "Local Node B script not found: ${PROJECT_DIR}/${NODE_B_SCRIPT}"

run_ssh \
    "test -f '$REMOTE_NODE_A_SCRIPT'" \
    || die "Remote Node A script not found: $REMOTE_NODE_A_SCRIPT"


# ------------------------------------------------------------
# 18. Validate cost files
# ------------------------------------------------------------

if [[ "$SELECTION_POLICY" == "cost" ]]; then

    [[ -f "$RECOMPUTE_COST_CSV" ]] ||
        die "Local recompute cost menu not found: $RECOMPUTE_COST_CSV"

    [[ -f "$RECOMPUTE_LAYER_PROFILE_CSV" ]] ||
        die "Local recompute profile not found: $RECOMPUTE_LAYER_PROFILE_CSV"

    run_ssh \
        "test -f '$REMOTE_RECOMPUTE_COST_CSV'" \
        || die "Remote cost CSV missing."

    run_ssh \
        "test -f '$REMOTE_RECOMPUTE_LAYER_PROFILE_CSV'" \
        || die "Remote recompute profile missing."

fi


# ------------------------------------------------------------
# 19. Create output directory
# ------------------------------------------------------------

mkdir -p "$RUN_DIR"

chown -R \
    "${SUDO_USER:-syu23}:${SUDO_USER:-syu23}" \
    "$RUN_DIR"

run_ssh \
    "mkdir -p '${REMOTE_PROJECT_DIR}/.splitmagic_remote'"


# ------------------------------------------------------------
# 20. Print experiment configuration
# ------------------------------------------------------------

log "============================================================"
log "Model             : ${MODEL}"
log "Dataset           : ${DATASET}"
log "Classes           : ${NUM_CLASSES}"
log "Batch size        : ${BATCH_SIZE}"
log
log "Node A            : ${REMOTE_TARGET}"
log "Node A threads    : ${NODE_A_THREADS}"
log "Node A script     : ${NODE_A_SCRIPT}"
log
log "Node B IP         : ${NODE_B_IP}"
log "Node B threads    : ${NODE_B_THREADS}"
log "Node B script     : ${NODE_B_SCRIPT}"
log
log "Network           : ${NETWORK_MBPS} Mbps"
log "Latency           : ${LATENCY_MS} ms"
log "MPS percent       : ${MPS_PERCENT}%"
log "Drop ratio        : ${DROP_RATIO}"
log "Selection policy  : ${SELECTION_POLICY}"
log "Run ID            : ${RUN_ID}"
log
log "Output directory  : ${RUN_DIR}"
log "============================================================"


# ------------------------------------------------------------
# 21. Remove stale JIN files
# ------------------------------------------------------------

log "Removing stale local JIN files..."

rm -f \
    /tmp/jin_execution_plan.tsv \
    /tmp/jin_template_plan.tsv \
    /tmp/jin_execution_plan_a.tsv \
    /tmp/jin_execution_plan_b.tsv


log "Removing stale remote JIN files..."

run_ssh \
    "rm -f \
        /tmp/jin_execution_plan.tsv \
        /tmp/jin_template_plan.tsv \
        /tmp/jin_execution_plan_a.tsv \
        /tmp/jin_execution_plan_b.tsv \
        '${REMOTE_PROJECT_DIR}/.splitmagic_remote/jin_template_plan_a.tsv'"


# ------------------------------------------------------------
# 22. Kill stale processes
# ------------------------------------------------------------

kill_remote_node_a

pkill -TERM -f "$NODE_B_SCRIPT" 2>/dev/null || true

sleep 1


# ------------------------------------------------------------
# 23. Apply network shaping
# ------------------------------------------------------------

if [[ "$ENABLE_TC" == "1" ]]; then

    log "Applying local B -> A traffic shaping..."

    tc qdisc del \
        dev "$LOCAL_IFACE" \
        root \
        2>/dev/null || true

    if [[ "$LATENCY_MS" != "0" || "$JITTER_MS" != "0" ]]; then

        tc qdisc replace \
            dev "$LOCAL_IFACE" \
            root \
            handle 1: \
            netem \
            delay "${LATENCY_MS}ms" "${JITTER_MS}ms" \
            rate "${NETWORK_MBPS}mbit" \
            limit "$QUEUE_LIMIT"

    else

        tc qdisc replace \
            dev "$LOCAL_IFACE" \
            root \
            tbf \
            rate "${NETWORK_MBPS}mbit" \
            burst 1mb \
            latency 100ms

    fi

    LOCAL_TC_APPLIED=1


    log "Applying remote A -> B traffic shaping..."

    run_ssh \
        "sudo tc qdisc del \
            dev '$REMOTE_IFACE' \
            root \
            2>/dev/null || true"


    if [[ "$LATENCY_MS" != "0" || "$JITTER_MS" != "0" ]]; then

        run_ssh \
            "sudo tc qdisc replace \
                dev '$REMOTE_IFACE' \
                root \
                handle 1: \
                netem \
                delay '${LATENCY_MS}ms' '${JITTER_MS}ms' \
                rate '${NETWORK_MBPS}mbit' \
                limit '$QUEUE_LIMIT'"

    else

        run_ssh \
            "sudo tc qdisc replace \
                dev '$REMOTE_IFACE' \
                root \
                tbf \
                rate '${NETWORK_MBPS}mbit' \
                burst 1mb \
                latency 100ms"

    fi

    REMOTE_TC_APPLIED=1

fi


# ------------------------------------------------------------
# 24. Save tc before statistics
# ------------------------------------------------------------

if [[ "$ENABLE_TC" == "1" ]]; then

    tc -s qdisc show \
        dev "$LOCAL_IFACE" \
        >"${RUN_DIR}/tc_b_before.txt" \
        || true

    run_ssh \
        "sudo tc -s qdisc show dev '$REMOTE_IFACE'" \
        >"${RUN_DIR}/tc_a_before.txt" \
        || true

fi


# ------------------------------------------------------------
# 25. Save experiment metadata
# ------------------------------------------------------------

cat >"${RUN_DIR}/experiment.env" <<EOF
MODEL=${MODEL}
DATASET=${DATASET}
NUM_CLASSES=${NUM_CLASSES}

PROJECT_DIR=${PROJECT_DIR}
REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR}

RUN_NAME=${RUN_NAME}
RUN_ID=${RUN_ID}

REMOTE_HOST=${REMOTE_HOST}
REMOTE_USER=${REMOTE_USER}

NODE_B_IP=${NODE_B_IP}
PORT=${PORT}

NODE_A_ENDPOINT=${NODE_A_ENDPOINT}
NODE_B_ENDPOINT=${NODE_B_ENDPOINT}

NODE_A_THREADS=${NODE_A_THREADS}
NODE_B_THREADS=${NODE_B_THREADS}

NETWORK_MBPS=${NETWORK_MBPS}
LATENCY_MS=${LATENCY_MS}
JITTER_MS=${JITTER_MS}

SELECTION_POLICY=${SELECTION_POLICY}
DROP_RATIO=${DROP_RATIO}

MPS_PERCENT=${MPS_PERCENT}

MAX_STEPS=${MAX_STEPS}
BATCH_SIZE=${BATCH_SIZE}

ENABLE_TC=${ENABLE_TC}
LOCAL_IFACE=${LOCAL_IFACE}
REMOTE_IFACE=${REMOTE_IFACE}

RECOMPUTE_COST_CSV=${RECOMPUTE_COST_CSV}
RECOMPUTE_LAYER_PROFILE_CSV=${RECOMPUTE_LAYER_PROFILE_CSV}
EOF


# ------------------------------------------------------------
# 26. Selection-policy environment
# ------------------------------------------------------------

# ------------------------------------------------------------
# 26. Start Node B under Nsight Systems
# ------------------------------------------------------------

log "Starting local Node B under Nsight Systems..."

setsid \
sudo -u "$RUN_USER" --preserve-env \
env \
    PATH="/usr/lib/nsight-systems/host-linux-x64:/home/syu23/miniconda3/envs/torch-build/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    LD_LIBRARY_PATH="/usr/lib/nsight-systems/host-linux-x64:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    OMP_NUM_THREADS="$NODE_B_THREADS" \
    MKL_NUM_THREADS="$NODE_B_THREADS" \
    JIN_CPU_THREADS="$NODE_B_THREADS" \
    JIN_RUN_ID="$RUN_ID" \
    JIN_EXPERIMENT_RUN_ID="$RUN_ID" \
    JIN_NETWORK_MBPS="$NETWORK_MBPS" \
    JIN_AUTO_DROP_RATIO="$DROP_RATIO" \
    JIN_SELECTION_POLICY="$SELECTION_POLICY" \
    JIN_MAX_STEPS="$MAX_STEPS" \
    JIN_BATCH_SIZE="$BATCH_SIZE" \
    JIN_NUM_CLASSES="$NUM_CLASSES" \
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
        '$PYTHON_BIN' '$NODE_B_SCRIPT'
" \
> "$NODE_B_LOG" 2>&1 &

NODE_B_PID=$!

log "Node B PID/PGID: ${NODE_B_PID}"


# ------------------------------------------------------------
# 28. Wait until Node B listens
# ------------------------------------------------------------

log "Waiting for Node B to listen on ${NODE_B_IP}:${PORT}..."

server_ready=0

for ((i = 0; i < NODE_B_START_TIMEOUT; i++)); do

    if ! kill -0 "$NODE_B_PID" 2>/dev/null; then

        die "Node B exited before becoming ready. Check ${NODE_B_LOG}"

    fi

    if ss -lnt |
        grep -q "${NODE_B_IP}:${PORT}"; then

        server_ready=1
        break

    fi

    # Also accept 0.0.0.0 binding
    if ss -lnt |
        grep -q "0.0.0.0:${PORT}"; then

        server_ready=1
        break

    fi

    sleep 1

done

[[ "$server_ready" -eq 1 ]] ||
    die "Node B did not listen within ${NODE_B_START_TIMEOUT}s."

log "Node B is ready"


# ------------------------------------------------------------
# 29. Check connectivity from Raspberry Pi
# ------------------------------------------------------------

log "Checking Node A -> Node B connectivity..."

run_ssh \
    "timeout 5 bash -c '</dev/tcp/${NODE_B_IP}/${PORT}'" \
    || die "Raspberry Pi cannot reach Node B at ${NODE_B_IP}:${PORT}"

log "Remote connection to Node B is reachable"


# ------------------------------------------------------------
# 30. Start remote Node A
# ------------------------------------------------------------

REMOTE_NODE_A_COMMAND="
set -Eeuo pipefail

cd '$REMOTE_PROJECT_DIR'

env \
    PYTHONPATH='$REMOTE_PYTHONPATH_VALUE' \
    LD_LIBRARY_PATH='/usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-sbsa-armv8:${LD_LIBRARY_PATH:-}' \
    OMP_NUM_THREADS='$NODE_A_THREADS' \
    MKL_NUM_THREADS='$NODE_A_THREADS' \
    CPU_THREADS='$NODE_A_THREADS' \
    JIN_CPU_THREADS='$NODE_A_THREADS' \
    JIN_RUN_ID='$RUN_ID' \
    JIN_EXPERIMENT_RUN_ID='$RUN_ID' \
    JIN_NETWORK_MBPS='$NETWORK_MBPS' \
    JIN_AUTO_DROP_RATIO='$DROP_RATIO' \
    JIN_MPS_PERCENT='$MPS_PERCENT' \
    JIN_SELECTION_POLICY='$SELECTION_POLICY' \
    JIN_NUM_CLASSES='$NUM_CLASSES' \
    JIN_TEMPLATE_PLAN_A_PATH='$TEMPLATE_PLAN_A_PATH' \
    JIN_MAX_STEPS='$MAX_STEPS' \
    JIN_BATCH_SIZE='$BATCH_SIZE' \
    JIN_ENDPOINT='$NODE_A_ENDPOINT' \
    REMOTE_USE_NSYS='$REMOTE_USE_NSYS' \
    REMOTE_PROJECT_DIR='$REMOTE_PROJECT_DIR' \
    RUN_NAME='$RUN_NAME' \
    REMOTE_PYTHON_BIN='$REMOTE_PYTHON_BIN' \
    REMOTE_NODE_A_SCRIPT='$REMOTE_NODE_A_SCRIPT' \
    bash -c '

if [[ "\$REMOTE_USE_NSYS" == "1" ]]; then

    exec nsys profile \
        --trace=nvtx,osrt \
        --sample=none \
        --force-overwrite=true \
        --output="\$REMOTE_PROJECT_DIR/.splitmagic_remote/node_a_\$RUN_NAME" \
        "\$REMOTE_PYTHON_BIN" \
        "\$REMOTE_NODE_A_SCRIPT"

else

    exec "\$REMOTE_PYTHON_BIN" \
        "\$REMOTE_NODE_A_SCRIPT"

fi
'
"

# echo "========================================"
# echo "[DEBUG][REMOTE_NODE_A_COMMAND]"
# printf '%s\n' "$REMOTE_NODE_A_COMMAND"
# echo "========================================"

setsid \
timeout \
    --signal=TERM \
    --kill-after=10s \
    "${EXPERIMENT_TIMEOUT}s" \
sudo -H -u "$RUN_USER" \
ssh "${SSH_OPTS[@]}" "$REMOTE_TARGET" \
    "$REMOTE_NODE_A_COMMAND" \
> "$NODE_A_LOG" 2>&1 &

NODE_A_PID=$!

log "Remote Node A SSH PID/PGID: ${NODE_A_PID}"


# ------------------------------------------------------------
# 31. Wait for remote Node A
# ------------------------------------------------------------

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

    die "Remote Node A failed with exit code ${NODE_A_EXIT}. Check ${NODE_A_LOG}"

fi

NODE_A_PID=""

log "Remote Node A finished"


# ------------------------------------------------------------
# 32. Wait for Node B
# ------------------------------------------------------------

log "Waiting for Node B to finish..."

set +e

timeout \
    --signal=TERM \
    --kill-after=10s \
    "${NODE_B_FINISH_TIMEOUT}s" \
bash -c '
    pid="$1"

    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
    done
' bash "$NODE_B_PID"

NODE_B_WAIT_EXIT=$?

set -e


if [[ "$NODE_B_WAIT_EXIT" -eq 124 ]]; then

    log "Node B did not stop automatically; terminating process group."

    kill_process_group "$NODE_B_PID"

fi

wait "$NODE_B_PID" 2>/dev/null || true

NODE_B_PID=""

log "Node B finished"


# ------------------------------------------------------------
# 33. Save final tc statistics
# ------------------------------------------------------------

if [[ "$ENABLE_TC" == "1" ]]; then

    tc -s qdisc show \
        dev "$LOCAL_IFACE" \
        >"${RUN_DIR}/tc_b_after.txt" \
        || true

    run_ssh \
        "sudo tc -s qdisc show dev '$REMOTE_IFACE'" \
        >"${RUN_DIR}/tc_a_after.txt" \
        || true

fi


# ------------------------------------------------------------
# 34. Optional remote Node A nsys report collection
# ------------------------------------------------------------

if [[ "$REMOTE_USE_NSYS" == "1" ]]; then

    log "Copying remote Node A qdstrm..."

    NODE_A_QDSTRM="${RUN_DIR}/node_a_${RUN_NAME}.qdstrm"
    NODE_A_REPORT="${RUN_DIR}/node_a_${RUN_NAME}.nsys-rep"

    run_scp \
        "${REMOTE_TARGET}:${REMOTE_PROJECT_DIR}/.splitmagic_remote/node_a_${RUN_NAME}.qdstrm" \
        "$NODE_A_QDSTRM" \
        || die "Failed to copy remote Node A qdstrm."

    log "Converting Node A qdstrm to nsys-rep..."

    /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
        -i "$NODE_A_QDSTRM" \
        -o "$NODE_A_REPORT" \
        -f \
        || die "Failed to convert Node A qdstrm."

    [[ -f "$NODE_A_REPORT" ]] \
        || die "Node A nsys-rep was not created: $NODE_A_REPORT"

fi


# ------------------------------------------------------------
# 35. Export local Node B Nsight statistics
# ------------------------------------------------------------

NODE_B_REPORT="${RUN_DIR}/node_b_${NETWORK_MBPS}_${MPS_PERCENT}_${DROP_RATIO}.nsys-rep"

export_nsys_csv() {

    local report="$1"
    local prefix="$2"

    if [[ ! -f "$report" ]]; then

        log "WARNING: Nsight report not found: $report"

        return 0

    fi

    log "Exporting Nsight statistics: $report"

    nsys stats \
        --report nvtxsum \
        --format csv \
        "$report" \
        >"${RUN_DIR}/${prefix}_nvtxsum.csv" \
        2>"${RUN_DIR}/${prefix}_nvtxsum_error.txt" \
        || log "WARNING: Failed to export ${prefix} NVTX summary."


    nsys stats \
        --report osrtsum \
        --format csv \
        "$report" \
        >"${RUN_DIR}/${prefix}_osrtsum.csv" \
        2>"${RUN_DIR}/${prefix}_osrtsum_error.txt" \
        || log "WARNING: Failed to export ${prefix} OS runtime summary."

}


export_node_b_cuda_csv() {

    local report="$1"

    if [[ ! -f "$report" ]]; then
        return 0
    fi


    nsys stats \
        --report cudaapisum \
        --format csv \
        "$report" \
        >"${RUN_DIR}/node_b_cudaapisum.csv" \
        2>"${RUN_DIR}/node_b_cudaapisum_error.txt" \
        || log "WARNING: Failed to export Node B CUDA API summary."


    nsys stats \
        --report gpukernsum \
        --format csv \
        "$report" \
        >"${RUN_DIR}/node_b_gpukernsum.csv" \
        2>"${RUN_DIR}/node_b_gpukernsum_error.txt" \
        || log "WARNING: Failed to export Node B GPU kernel summary."

}


export_nsys_csv \
    "$NODE_B_REPORT" \
    "node_b"

export_node_b_cuda_csv \
    "$NODE_B_REPORT"


# ------------------------------------------------------------
# 36. Optional remote Node A NVTX export
# ------------------------------------------------------------

if [[ "$REMOTE_USE_NSYS" == "1" ]]; then

    if [[ -f "$NODE_A_REPORT" ]]; then

        export_nsys_csv \
            "$NODE_A_REPORT" \
            "node_a"

    else

        log "WARNING: Node A nsys report not found: $NODE_A_REPORT"

    fi

fi


# ------------------------------------------------------------
# 37. Remove traffic shaping
# ------------------------------------------------------------

remove_local_tc
remove_remote_tc


# ------------------------------------------------------------
# 38. Summary
# ------------------------------------------------------------

log "============================================================"
log "Experiment complete"
log
log "Model             : ${MODEL}"
log "Dataset           : ${DATASET}"
log
log "Node A            : ${REMOTE_TARGET}"
log "Node A log        : ${NODE_A_LOG}"
log
log "Node B            : local GPU"
log "Node B log        : ${NODE_B_LOG}"
log
log "Results directory : ${RUN_DIR}"

if [[ -f "$NODE_B_REPORT" ]]; then

    log "Node B NSYS       : ${NODE_B_REPORT}"

fi

log "============================================================"


# ------------------------------------------------------------
# 39. Successful exit
# ------------------------------------------------------------

exit 0