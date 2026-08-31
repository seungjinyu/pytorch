#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# SplitMagic remote sweep
#
# Sweep:
#   model × network × drop ratio × MPS percentage × repeat
#
# Node A : Remote CPU machine (e.g., Raspberry Pi 5)
# Node B : Local GPU machine
#
# Each combination invokes:
#   run_remote_nsys.sh
#
# Usage example:
#
#   chmod +x run_all_remote_nsys.sh
#   chmod +x run_remote_nsys.sh
#
#   REMOTE_HOST=192.168.1.50 \
#   NODE_B_IP=192.168.1.40 \
#   LOCAL_IFACE=enp3s0 \
#   REMOTE_IFACE=eth0 \
#   ./run_all_remote_nsys.sh
#
# ============================================================


# ------------------------------------------------------------
# 1. Project configuration
# ------------------------------------------------------------

PROJECT_DIR="${PROJECT_DIR:-/home/syu23/seungjin/pytorch/splitmagic}"

RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/run_remote_nsys.sh}"


# ------------------------------------------------------------
# 2. Remote Node A configuration
# ------------------------------------------------------------

REMOTE_HOST="${REMOTE_HOST:-10.32.126.244}"

REMOTE_USER="${REMOTE_USER:-ubuntu}"

REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/ubuntu/splitmagic_project}"

REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-/home/ubuntu/venv/splitmagic/bin/python}"

SSH_PORT="${SSH_PORT:-22}"


# ------------------------------------------------------------
# 3. Local Node B network configuration
# ------------------------------------------------------------

# Local GPU node IP reachable from Raspberry Pi.
#
# Example:
# NODE_B_IP=192.168.1.40
NODE_B_IP="${NODE_B_IP:-10.32.137.57}"

PORT="${PORT:-5556}"


# ------------------------------------------------------------
# 4. Network interfaces
# ------------------------------------------------------------

# Local GPU machine Ethernet interface.
#
# Example:
#   enp3s0
#
LOCAL_IFACE="${LOCAL_IFACE:-enp3s0}"

# Raspberry Pi Ethernet interface.
#
# Usually:
#   eth0
#
REMOTE_IFACE="${REMOTE_IFACE:-eth0}"

# 1 = use tc bandwidth shaping
# 0 = use physical network as-is
ENABLE_TC="${ENABLE_TC:-1}"


# ------------------------------------------------------------
# 5. Models
# ------------------------------------------------------------

MODELS=(
    resnet18
    resnet18_imagenet
)


# ------------------------------------------------------------
# 6. Network bandwidth sweep
# ------------------------------------------------------------

NETWORKS=(
    1000
    # 500
    # 200
    # 100
    # 50
)


# ------------------------------------------------------------
# 7. Drop ratio sweep
# ------------------------------------------------------------

MAX_RATIOS=(
    # 0.0
    # 0.1
    # 0.2
    # 0.3
    # 0.4
    # 0.5
    # 0.6
    # 0.7
    # 0.8
    # 0.9
    # 0.95
    # 0.96
    # 0.97
    # 0.98
    # 0.99
    1.0
)


# ------------------------------------------------------------
# 8. GPU MPS sweep
# ------------------------------------------------------------

MPS_PERCENTAGES=(
    # 100
    # 90
    # 80
    # 70
    # 60
    # 50
    # 40
    # 30
    # 20
    # 10
    25
)


# ------------------------------------------------------------
# 9. Experiment configuration
# ------------------------------------------------------------

REPEATS="${REPEATS:-1}"

MAX_STEPS="${MAX_STEPS:-50}"

LATENCY_MS="${LATENCY_MS:-0}"

JITTER_MS="${JITTER_MS:-0}"

BATCH_SIZE="${BATCH_SIZE:-32}"

SELECTION_POLICY="${SELECTION_POLICY:-ratio}"


# ------------------------------------------------------------
# 10. CPU thread configuration
# ------------------------------------------------------------

# Raspberry Pi / remote Node A
NODE_A_THREADS="${NODE_A_THREADS:-4}"

# Local Node B CPU-side PyTorch threads
NODE_B_THREADS="${NODE_B_THREADS:-1}"


# ------------------------------------------------------------
# 11. Nsight configuration
# ------------------------------------------------------------

# Normally keep disabled for Raspberry Pi.
#
# 0:
#   Node A runs normally over SSH.
#
# 1:
#   Node A also runs under nsys.
#
REMOTE_USE_NSYS="${REMOTE_USE_NSYS:-1}"


# ------------------------------------------------------------
# 12. Timeouts
# ------------------------------------------------------------

NODE_B_START_TIMEOUT="${NODE_B_START_TIMEOUT:-60}"

EXPERIMENT_TIMEOUT="${EXPERIMENT_TIMEOUT:-1200}"

NODE_B_FINISH_TIMEOUT="${NODE_B_FINISH_TIMEOUT:-120}"


# ------------------------------------------------------------
# 13. Sweep delay
# ------------------------------------------------------------

SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-3}"


# ------------------------------------------------------------
# 14. Helpers
# ------------------------------------------------------------

log() {
    printf '[REMOTE-SWEEP][%s] %s\n' \
        "$(date '+%H:%M:%S')" \
        "$*"
}


die() {
    printf '[REMOTE-SWEEP][ERROR] %s\n' \
        "$*" \
        >&2

    exit 1
}


# ------------------------------------------------------------
# 15. Validation
# ------------------------------------------------------------

[[ -d "$PROJECT_DIR" ]] ||
    die "PROJECT_DIR does not exist: $PROJECT_DIR"

[[ -x "$RUN_SCRIPT" ]] ||
    die "Run script is not executable: $RUN_SCRIPT"

[[ -n "$REMOTE_HOST" ]] ||
    die "REMOTE_HOST is required."

[[ -n "$NODE_B_IP" ]] ||
    die "NODE_B_IP is required."

if [[ "$ENABLE_TC" == "1" ]]; then

    [[ -n "$LOCAL_IFACE" ]] ||
        die "LOCAL_IFACE is required when ENABLE_TC=1."

fi


# ------------------------------------------------------------
# 16. Move to project directory
# ------------------------------------------------------------

cd "$PROJECT_DIR"


# ------------------------------------------------------------
# 17. Calculate total runs
# ------------------------------------------------------------

TOTAL_RUNS=$(( \
    ${#MODELS[@]} * \
    ${#NETWORKS[@]} * \
    ${#MAX_RATIOS[@]} * \
    ${#MPS_PERCENTAGES[@]} * \
    REPEATS \
))

CURRENT_RUN=0


# ------------------------------------------------------------
# 18. Print sweep configuration
# ------------------------------------------------------------

log "============================================================"
log "Starting SplitMagic remote sweep"
log
log "Project directory   : ${PROJECT_DIR}"
log "Run script          : ${RUN_SCRIPT}"
log
log "Remote host         : ${REMOTE_HOST}"
log "Remote user         : ${REMOTE_USER}"
log "Remote project      : ${REMOTE_PROJECT_DIR}"
log
log "Node B IP           : ${NODE_B_IP}"
log "Port                : ${PORT}"
log
log "Local interface     : ${LOCAL_IFACE:-not-used}"
log "Remote interface    : ${REMOTE_IFACE}"
log "TC enabled          : ${ENABLE_TC}"
log
log "Models              : ${MODELS[*]}"
log "Networks            : ${NETWORKS[*]}"
log "Drop ratios         : ${MAX_RATIOS[*]}"
log "MPS percentages     : ${MPS_PERCENTAGES[*]}"
log
log "Batch size          : ${BATCH_SIZE}"
log "Selection policy    : ${SELECTION_POLICY}"
log "Node A threads      : ${NODE_A_THREADS}"
log "Node B threads      : ${NODE_B_THREADS}"
log
log "Max steps           : ${MAX_STEPS}"
log "Repeats             : ${REPEATS}"
log "Total runs          : ${TOTAL_RUNS}"
log "============================================================"


# ------------------------------------------------------------
# 19. Sweep
# ------------------------------------------------------------

for model in "${MODELS[@]}"; do

    for network in "${NETWORKS[@]}"; do

        for max_ratio in "${MAX_RATIOS[@]}"; do

            for mps in "${MPS_PERCENTAGES[@]}"; do

                for ((run_id = 0; run_id < REPEATS; run_id++)); do

                    CURRENT_RUN=$((CURRENT_RUN + 1))

                    log "============================================================"
                    log "Run ${CURRENT_RUN}/${TOTAL_RUNS}"
                    log
                    log "model=${model}"
                    log "network=${network} Mbps"
                    log "drop_ratio=${max_ratio}"
                    log "mps=${mps}%"
                    log "batch_size=${BATCH_SIZE}"
                    log "node_a_threads=${NODE_A_THREADS}"
                    log "node_b_threads=${NODE_B_THREADS}"
                    log "run_id=${run_id}"
                    log "============================================================"

                    sudo -E \
                        PROJECT_DIR="$PROJECT_DIR" \
                        MODEL="$model" \
                        REMOTE_HOST="$REMOTE_HOST" \
                        REMOTE_USER="$REMOTE_USER" \
                        REMOTE_PROJECT_DIR="$REMOTE_PROJECT_DIR" \
                        REMOTE_PYTHON_BIN="$REMOTE_PYTHON_BIN" \
                        SSH_PORT="$SSH_PORT" \
                        NODE_B_IP="$NODE_B_IP" \
                        PORT="$PORT" \
                        LOCAL_IFACE="$LOCAL_IFACE" \
                        REMOTE_IFACE="$REMOTE_IFACE" \
                        ENABLE_TC="$ENABLE_TC" \
                        NETWORK_MBPS="$network" \
                        LATENCY_MS="$LATENCY_MS" \
                        JITTER_MS="$JITTER_MS" \
                        DROP_RATIO="$max_ratio" \
                        SELECTION_POLICY="$SELECTION_POLICY" \
                        MPS_PERCENT="$mps" \
                        RUN_ID="$run_id" \
                        JIN_BATCH_SIZE="$BATCH_SIZE" \
                        NODE_A_THREADS="$NODE_A_THREADS" \
                        NODE_B_THREADS="$NODE_B_THREADS" \
                        MAX_STEPS="$MAX_STEPS" \
                        REMOTE_USE_NSYS="$REMOTE_USE_NSYS" \
                        NODE_B_START_TIMEOUT="$NODE_B_START_TIMEOUT" \
                        EXPERIMENT_TIMEOUT="$EXPERIMENT_TIMEOUT" \
                        NODE_B_FINISH_TIMEOUT="$NODE_B_FINISH_TIMEOUT" \
                        "$RUN_SCRIPT"

                    log "Finished run ${CURRENT_RUN}/${TOTAL_RUNS}"

                    if [[ "$CURRENT_RUN" -lt "$TOTAL_RUNS" ]]; then

                        log "Sleeping ${SLEEP_BETWEEN_RUNS}s before next run..."

                        sleep "$SLEEP_BETWEEN_RUNS"

                    fi

                done

            done

        done

    done

done


# ------------------------------------------------------------
# 20. Summary
# ------------------------------------------------------------

log "============================================================"
log "All remote experiments completed"
log "Completed runs: ${CURRENT_RUN}/${TOTAL_RUNS}"
log "============================================================"

exit 0