#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# CPU Sweep:
#   network × drop ratio × CPU threads × repeat
#
# Each combination invokes:
#   run_local_nsys_cpu.sh
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-/home/syu23/seungjin/pytorch/splitmagic}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/run_local_nsys_cpu.sh}"

# ------------------------------------------------------------
# Sweep configuration
# ------------------------------------------------------------

NETWORKS=(
    1000
    # 500
    # 200
    # 100
    # 50
)

DROP_RATIOS=(
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

CPU_THREAD_COUNTS=(
    1
    # 2
    # 4
    # 6
)

REPEATS="${REPEATS:-1}"
MAX_STEPS="${MAX_STEPS:-10}"
LATENCY_MS="${LATENCY_MS:-0}"
MODEL="${MODEL:-resnet18_imagenet}"
BATCH_SIZE="${BATCH_SIZE:-32}"

SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-3}"

log() {
    printf '[CPU_SWEEP][%s] %s\n' \
        "$(date '+%H:%M:%S')" \
        "$*"
}

die() {
    printf '[CPU_SWEEP][ERROR] %s\n' "$*" >&2
    exit 1
}

[[ -x "${RUN_SCRIPT}" ]] ||
    die "Run script is not executable: ${RUN_SCRIPT}"

cd "${PROJECT_DIR}"

TOTAL_RUNS=$(( \
    ${#NETWORKS[@]} * \
    ${#DROP_RATIOS[@]} * \
    ${#CPU_THREAD_COUNTS[@]} * \
    REPEATS \
))

CURRENT_RUN=0

log "Starting SplitMagic CPU sweep"
log "Networks        : ${NETWORKS[*]}"
log "Drop ratios     : ${DROP_RATIOS[*]}"
log "CPU threads     : ${CPU_THREAD_COUNTS[*]}"
log "Repeats         : ${REPEATS}"
log "Max steps       : ${MAX_STEPS}"
log "Latency         : ${LATENCY_MS} ms"
log "Model           : ${MODEL}"
log "Batch size      : ${BATCH_SIZE}"
log "Total runs      : ${TOTAL_RUNS}"

for network in "${NETWORKS[@]}"; do
    for ratio in "${DROP_RATIOS[@]}"; do
        for cpu_threads in "${CPU_THREAD_COUNTS[@]}"; do
            for ((run_id = 0; run_id < REPEATS; run_id++)); do
                CURRENT_RUN=$((CURRENT_RUN + 1))

                log "============================================================"
                log "Run ${CURRENT_RUN}/${TOTAL_RUNS}"
                log "network=${network} Mbps"
                log "drop_ratio=${ratio}"
                log "cpu_threads=${cpu_threads}"
                log "run_id=${run_id}"
                log "============================================================"

                sudo -E \
                    PROJECT_DIR="${PROJECT_DIR}" \
                    NETWORK_MBPS="${network}" \
                    LATENCY_MS="${LATENCY_MS}" \
                    DROP_RATIO="${ratio}" \
                    CPU_THREADS="${cpu_threads}" \
                    RUN_ID="${run_id}" \
                    MAX_STEPS="${MAX_STEPS}" \
                    MODEL="${MODEL}" \
                    BATCH_SIZE="${BATCH_SIZE}" \
                    "${RUN_SCRIPT}"

                log "Finished run ${CURRENT_RUN}/${TOTAL_RUNS}"

                sleep "${SLEEP_BETWEEN_RUNS}"
            done
        done
    done
done

log "All CPU experiments completed"