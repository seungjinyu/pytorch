#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# SplitMagic local CPU + Nsight Systems experiment
#
# Required/optional environment variables:
#
#   PROJECT_DIR
#   NETWORK_MBPS
#   LATENCY_MS
#   DROP_RATIO
#   CPU_THREADS
#   RUN_ID
#   MAX_STEPS
#   MODEL
#
# Example:
#
#   sudo -E \
#     NETWORK_MBPS=1000 \
#     LATENCY_MS=0 \
#     DROP_RATIO=0.99 \
#     CPU_THREADS=1 \
#     RUN_ID=0 \
#     MAX_STEPS=5 \
#     MODEL=resnet18 \
#     ./run_local_nsys_cpu.sh
# ============================================================


# ------------------------------------------------------------
# Basic configuration
# ------------------------------------------------------------

PROJECT_DIR="${PROJECT_DIR:-/home/syu23/seungjin/pytorch/splitmagic}"

PYTHON_BIN="${PYTHON_BIN:-/home/syu23/miniconda3/envs/torch-build/bin/python3}"

NETWORK_MBPS="${NETWORK_MBPS:-1000}"
LATENCY_MS="${LATENCY_MS:-0}"
DROP_RATIO="${DROP_RATIO:-0.99}"
CPU_THREADS="${CPU_THREADS:-1}"
RUN_ID="${RUN_ID:-0}"
MAX_STEPS="${MAX_STEPS:-5}"
MODEL="${MODEL:-resnet18}"

PORT="${PORT:-5555}"

NODE_A_TIMEOUT_SEC="${NODE_A_TIMEOUT_SEC:-600}"
NODE_B_READY_TIMEOUT_SEC="${NODE_B_READY_TIMEOUT_SEC:-60}"

NS_A="${NS_A:-jin_cpu_ns_a}"
NS_B="${NS_B:-jin_cpu_ns_b}"

VETH_A="${VETH_A:-jin_cpu_veth_a}"
VETH_B="${VETH_B:-jin_cpu_veth_b}"

IP_A="${IP_A:-10.10.0.1}"
IP_B="${IP_B:-10.10.0.2}"
SUBNET_CIDR="${SUBNET_CIDR:-24}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

OUTPUT_DIR="${PROJECT_DIR}/nsys_results/${TIMESTAMP}_${MODEL}_cpu_${NETWORK_MBPS}mbps_${LATENCY_MS}ms_ratio${DROP_RATIO}_threads${CPU_THREADS}_run${RUN_ID}"

NODE_A_LOG="${OUTPUT_DIR}/node_a.log"
NODE_B_LOG="${OUTPUT_DIR}/node_b.log"

NODE_A_CSV="${OUTPUT_DIR}/node_a_${MODEL}_cpu.csv"
NODE_B_CSV="${OUTPUT_DIR}/node_b_${MODEL}_cpu.csv"

EXPERIMENT_CSV="${OUTPUT_DIR}/drop_ratio_experiment_${MODEL}_cpu.csv"

TEMPLATE_PLAN_A="${OUTPUT_DIR}/jin_template_plan_a.tsv"
TEMPLATE_PLAN_B="${OUTPUT_DIR}/jin_template_plan_b.tsv"
EXECUTION_PLAN="${OUTPUT_DIR}/jin_execution_plan.tsv"
ALIAS_PATH="${OUTPUT_DIR}/jin_payload.alias"

NSYS_BASE="${OUTPUT_DIR}/node_b_${NETWORK_MBPS}_cpu_t${CPU_THREADS}_${DROP_RATIO}"


# ------------------------------------------------------------
# Model-specific scripts
# ------------------------------------------------------------

case "${MODEL}" in
    resnet18)
        NODE_A_SCRIPT="tests/test_node_a_resnet18.py"
        NODE_B_SCRIPT="tests/test_node_b_resnet18.py"
        ;;

    vgg|vgg11bn)
        NODE_A_SCRIPT="tests/test_node_a_vgg.py"
        NODE_B_SCRIPT="tests/test_node_b_vgg.py"
        ;;

    *)
        echo "[ERROR] Unsupported MODEL=${MODEL}" >&2
        echo "[ERROR] Expected resnet18, vgg, or vgg11bn" >&2
        exit 1
        ;;
esac


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

log() {
    printf '[RUN][%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf '[RUN][ERROR] %s\n' "$*" >&2
    exit 1
}


# ------------------------------------------------------------
# Runtime state
# ------------------------------------------------------------

NODE_A_PID=""
NODE_B_PID=""


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

stop_process() {
    local pid="${1:-}"

    if [[ -z "${pid}" ]]; then
        return
    fi

    if kill -0 "${pid}" 2>/dev/null; then
        log "Stopping PID ${pid}"

        kill -TERM "-${pid}" 2>/dev/null || \
        kill -TERM "${pid}" 2>/dev/null || true

        sleep 2

        kill -KILL "-${pid}" 2>/dev/null || \
        kill -KILL "${pid}" 2>/dev/null || true
    fi
}


cleanup_network() {
    ip netns del "${NS_A}" 2>/dev/null || true
    ip netns del "${NS_B}" 2>/dev/null || true

    ip link del "${VETH_A}" 2>/dev/null || true
}


cleanup() {
    local status=$?

    set +e

    log "Cleaning up..."

    stop_process "${NODE_A_PID}"
    stop_process "${NODE_B_PID}"

    cleanup_network

    log "Cleanup complete"

    exit "${status}"
}

trap cleanup EXIT INT TERM


# ------------------------------------------------------------
# Preconditions
# ------------------------------------------------------------

[[ "$(id -u)" -eq 0 ]] ||
    die "This script must be run with sudo/root."

[[ -d "${PROJECT_DIR}" ]] ||
    die "Project directory not found: ${PROJECT_DIR}"

[[ -x "${PYTHON_BIN}" ]] ||
    die "Python binary not executable: ${PYTHON_BIN}"

command -v nsys >/dev/null 2>&1 ||
    die "nsys command not found"

command -v ip >/dev/null 2>&1 ||
    die "ip command not found"

command -v tc >/dev/null 2>&1 ||
    die "tc command not found"

[[ -f "${PROJECT_DIR}/${NODE_A_SCRIPT}" ]] ||
    die "Node A script not found: ${NODE_A_SCRIPT}"

[[ -f "${PROJECT_DIR}/${NODE_B_SCRIPT}" ]] ||
    die "Node B script not found: ${NODE_B_SCRIPT}"


# ------------------------------------------------------------
# Project environment
# ------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/..:/home/syu23/torchvision-0.17:${PYTHONPATH:-}"

log "Project directory : ${PROJECT_DIR}"
log "Output directory  : ${OUTPUT_DIR}"
log "Python binary     : ${PYTHON_BIN}"
log "Model             : ${MODEL}"
log "Network           : ${NETWORK_MBPS} Mbps"
log "Latency           : ${LATENCY_MS} ms"
log "Drop ratio        : ${DROP_RATIO}"
log "CPU threads       : ${CPU_THREADS}"
log "Run ID            : ${RUN_ID}"
log "Max steps         : ${MAX_STEPS}"


# ------------------------------------------------------------
# Remove stale processes/files
# ------------------------------------------------------------

log "Stopping stale Node A/Node B processes..."

pkill -f "${NODE_A_SCRIPT}" 2>/dev/null || true
pkill -f "${NODE_B_SCRIPT}" 2>/dev/null || true

sleep 2

rm -f \
    /tmp/jin_template_plan.tsv \
    /tmp/jin_template_plan_a.tsv \
    /tmp/jin_execution_plan.tsv \
    /tmp/jin_payload_recv.bin \
    /tmp/jin_payload_recv.bin.alias \
    "${TEMPLATE_PLAN_A}" \
    "${TEMPLATE_PLAN_B}" \
    "${EXECUTION_PLAN}" \
    "${ALIAS_PATH}"


# ------------------------------------------------------------
# Create network namespaces
# ------------------------------------------------------------

log "Creating network namespaces..."

cleanup_network

ip netns add "${NS_A}"
ip netns add "${NS_B}"

ip link add "${VETH_A}" type veth peer name "${VETH_B}"

ip link set "${VETH_A}" netns "${NS_A}"
ip link set "${VETH_B}" netns "${NS_B}"

ip -n "${NS_A}" addr add "${IP_A}/${SUBNET_CIDR}" dev "${VETH_A}"
ip -n "${NS_B}" addr add "${IP_B}/${SUBNET_CIDR}" dev "${VETH_B}"

ip -n "${NS_A}" link set lo up
ip -n "${NS_B}" link set lo up

ip -n "${NS_A}" link set "${VETH_A}" up
ip -n "${NS_B}" link set "${VETH_B}" up


# ------------------------------------------------------------
# Apply network shaping
# ------------------------------------------------------------

log "Applying traffic control..."

# Remove any existing qdisc.
ip netns exec "${NS_A}" \
    tc qdisc del dev "${VETH_A}" root 2>/dev/null || true

ip netns exec "${NS_B}" \
    tc qdisc del dev "${VETH_B}" root 2>/dev/null || true


# Node A -> Node B bandwidth/latency.
ip netns exec "${NS_A}" \
    tc qdisc add dev "${VETH_A}" root handle 1: \
    tbf \
    rate "${NETWORK_MBPS}mbit" \
    burst 1mb \
    latency 500ms

if [[ "${LATENCY_MS}" != "0" ]]; then
    ip netns exec "${NS_A}" \
        tc qdisc add dev "${VETH_A}" parent 1:1 handle 10: \
        netem delay "${LATENCY_MS}ms"
fi


# Node B -> Node A bandwidth/latency.
ip netns exec "${NS_B}" \
    tc qdisc add dev "${VETH_B}" root handle 1: \
    tbf \
    rate "${NETWORK_MBPS}mbit" \
    burst 1mb \
    latency 500ms

if [[ "${LATENCY_MS}" != "0" ]]; then
    ip netns exec "${NS_B}" \
        tc qdisc add dev "${VETH_B}" parent 1:1 handle 10: \
        netem delay "${LATENCY_MS}ms"
fi


# ------------------------------------------------------------
# Verify network
# ------------------------------------------------------------

log "Checking virtual network..."

if ! ip netns exec "${NS_A}" \
    ping -c 1 -W 2 "${IP_B}" >/dev/null
then
    die "Virtual network ping failed: ${IP_A} -> ${IP_B}"
fi

log "Virtual network is reachable"


# ------------------------------------------------------------
# Shared CPU settings
# ------------------------------------------------------------

CPU_ENV=(
    "JIN_NODE_B_DEVICE=cpu"
    "CPU_THREADS=${CPU_THREADS}"
    "OMP_NUM_THREADS=${CPU_THREADS}"
    "MKL_NUM_THREADS=${CPU_THREADS}"
    "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
    "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
    "VECLIB_MAXIMUM_THREADS=${CPU_THREADS}"
)


# ------------------------------------------------------------
# Start Node B under Nsight Systems
# ------------------------------------------------------------

log "Starting Node B on CPU under Nsight Systems..."

setsid ip netns exec "${NS_B}" \
    env \
    PYTHONPATH="${PYTHONPATH}" \
    JIN_ENDPOINT="tcp://*:${PORT}" \
    JIN_NODE_B_DEVICE="cpu" \
    CPU_THREADS="${CPU_THREADS}" \
    OMP_NUM_THREADS="${CPU_THREADS}" \
    MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" \
    NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
    VECLIB_MAXIMUM_THREADS="${CPU_THREADS}" \
    JIN_MAX_STEPS="${MAX_STEPS}" \
    JIN_NETWORK_MBPS="${NETWORK_MBPS}" \
    JIN_AUTO_DROP_RATIO="${DROP_RATIO}" \
    JIN_RUN_ID="${RUN_ID}" \
    JIN_EXPERIMENT_RUN_ID="${RUN_ID}" \
    JIN_BATCH_SIZE="32" \
    JIN_TEMPLATE_PLAN_PATH="${TEMPLATE_PLAN_B}" \
    JIN_EXECUTION_PLAN_PATH="${EXECUTION_PLAN}" \
    JIN_ALIAS_PATH="${ALIAS_PATH}" \
    JIN_RECOMPUTE_EXPERIMENT_CSV="${OUTPUT_DIR}/recompute_cost_experiments_cpu.csv" \
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --force-overwrite=true \
        --output="${NSYS_BASE}" \
        "${PYTHON_BIN}" -u "${NODE_B_SCRIPT}" \
    >"${NODE_B_LOG}" 2>&1 &

NODE_B_PID=$!

log "Node B PID/PGID: ${NODE_B_PID}"


# ------------------------------------------------------------
# Wait for Node B
# ------------------------------------------------------------

log "Waiting for Node B to listen on ${IP_B}:${PORT}..."

node_b_ready=0

for _ in $(seq 1 "${NODE_B_READY_TIMEOUT_SEC}"); do
    if grep -q '\[Node B\] listening' "${NODE_B_LOG}" 2>/dev/null; then
        node_b_ready=1
        break
    fi

    if ! kill -0 "${NODE_B_PID}" 2>/dev/null; then
        echo
        echo "================ Node B log ================"
        tail -200 "${NODE_B_LOG}" || true
        echo "============================================"
        die "Node B exited before becoming ready."
    fi

    sleep 1
done

if [[ "${node_b_ready}" -ne 1 ]]; then
    echo
    echo "================ Node B log ================"
    tail -200 "${NODE_B_LOG}" || true
    echo "============================================"
    die "Node B did not become ready."
fi

log "Node B is ready"


# ------------------------------------------------------------
# Start Node A
# ------------------------------------------------------------

log "Starting Node A..."

setsid ip netns exec "${NS_A}" \
    env \
    PYTHONPATH="${PYTHONPATH}" \
    JIN_ENDPOINT="tcp://${IP_B}:${PORT}" \
    JIN_MAX_STEPS="${MAX_STEPS}" \
    JIN_NETWORK_MBPS="${NETWORK_MBPS}" \
    JIN_AUTO_DROP_RATIO="${DROP_RATIO}" \
    JIN_RUN_ID="${RUN_ID}" \
    JIN_EXPERIMENT_RUN_ID="${RUN_ID}" \
    JIN_TEMPLATE_PLAN_A_PATH="${TEMPLATE_PLAN_A}" \
    JIN_EXPERIMENT_CSV="${EXPERIMENT_CSV}" \
    CPU_THREADS="1" \
    OMP_NUM_THREADS="1" \
    MKL_NUM_THREADS="1" \
    OPENBLAS_NUM_THREADS="1" \
    "${PYTHON_BIN}" -u "${NODE_A_SCRIPT}" \
    >"${NODE_A_LOG}" 2>&1 &

NODE_A_PID=$!

log "Node A PID/PGID: ${NODE_A_PID}"


# ------------------------------------------------------------
# Wait for Node A with timeout
# ------------------------------------------------------------

start_epoch="$(date +%s)"

while kill -0 "${NODE_A_PID}" 2>/dev/null; do
    now_epoch="$(date +%s)"
    elapsed=$((now_epoch - start_epoch))

    if (( elapsed >= NODE_A_TIMEOUT_SEC )); then
        echo
        echo "================ Node A log ================"
        tail -200 "${NODE_A_LOG}" || true
        echo "================ Node B log ================"
        tail -200 "${NODE_B_LOG}" || true
        echo "============================================"

        die "Node A timed out after ${NODE_A_TIMEOUT_SEC}s."
    fi

    sleep 1
done


# Collect Node A exit status.
set +e
wait "${NODE_A_PID}"
NODE_A_STATUS=$?
set -e

NODE_A_PID=""

if [[ "${NODE_A_STATUS}" -ne 0 ]]; then
    echo
    echo "================ Node A log ================"
    tail -200 "${NODE_A_LOG}" || true
    echo "================ Node B log ================"
    tail -200 "${NODE_B_LOG}" || true
    echo "============================================"

    die "Node A failed with status ${NODE_A_STATUS}."
fi

log "Node A finished successfully"


# ------------------------------------------------------------
# Wait for Node B / Nsight profiler
# ------------------------------------------------------------

log "Waiting for Node B profiler to finish..."

set +e
wait "${NODE_B_PID}"
NODE_B_STATUS=$?
set -e

NODE_B_PID=""

if [[ "${NODE_B_STATUS}" -ne 0 ]]; then
    echo
    echo "================ Node B log ================"
    tail -300 "${NODE_B_LOG}" || true
    echo "============================================"

    die "Node B/Nsight failed with status ${NODE_B_STATUS}."
fi


# ------------------------------------------------------------
# Locate/convert Nsight report
# ------------------------------------------------------------

NSYS_REPORT="${NSYS_BASE}.nsys-rep"
QDSTRM_REPORT="${NSYS_BASE}.qdstrm"

if [[ ! -f "${NSYS_REPORT}" && -f "${QDSTRM_REPORT}" ]]; then
    IMPORTER="/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"

    if [[ ! -x "${IMPORTER}" ]]; then
        IMPORTER="/usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64/QdstrmImporter"
    fi

    if [[ -x "${IMPORTER}" ]]; then
        log "Converting QDSTRM to NSYS-REP..."

        "${IMPORTER}" \
            "${QDSTRM_REPORT}" \
            --output-file "${NSYS_REPORT}" \
            >>"${NODE_B_LOG}" 2>&1 || true
    else
        log "WARNING: QdstrmImporter not found"
    fi
fi


# ------------------------------------------------------------
# Export NVTX summary
# ------------------------------------------------------------

if [[ -f "${NSYS_REPORT}" ]]; then
    NVTX_CSV="${OUTPUT_DIR}/node_b_${NETWORK_MBPS}_cpu_t${CPU_THREADS}_${DROP_RATIO}_nvtxsum.csv"

    log "Exporting NVTX summary..."

    nsys stats \
        --report nvtxsum \
        --format csv \
        --output "${NVTX_CSV}" \
        "${NSYS_REPORT}" \
        >>"${NODE_B_LOG}" 2>&1 || true

    log "NSYS report : ${NSYS_REPORT}"
    log "NVTX CSV    : ${NVTX_CSV}"
else
    log "WARNING: NSYS report was not generated."
    log "Expected: ${NSYS_REPORT}"
    log "QDSTRM  : ${QDSTRM_REPORT}"
fi


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

if grep -q 'Traceback' "${NODE_A_LOG}"; then
    die "Traceback found in Node A log."
fi

if grep -q 'Traceback' "${NODE_B_LOG}"; then
    die "Traceback found in Node B log."
fi

log "CPU experiment completed successfully"
log "Node A log: ${NODE_A_LOG}"
log "Node B log: ${NODE_B_LOG}"
log "Output dir: ${OUTPUT_DIR}"

# Restore ownership
if [[ -n "${SUDO_USER:-}" ]]; then
    log "Restoring ownership to ${SUDO_USER}"

    chown -R \
        "${SUDO_USER}:${SUDO_USER}" \
        "${OUTPUT_DIR}"
fi