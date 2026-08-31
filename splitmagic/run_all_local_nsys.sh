#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Sweep:
#   network × drop ratio × MPS percentage × repeat
#
# Each combination invokes:
#   run_local_nsys.sh
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-/home/syu23/seungjin/pytorch/splitmagic}"
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_DIR}/run_local_nsys.sh}"

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

MAX_RATIOS=(
    # 0.0
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

REPEATS="${REPEATS:-1}"
MAX_STEPS="${MAX_STEPS:-50}"
LATENCY_MS="${LATENCY_MS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# resnet18 , vgg, resnet18_imagenet
# MODEL="${MODEL:-resnet18}"
MODEL="${MODEL:-resnet18}"

# 실험 사이 GPU/프로세스 정리 대기
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-3}"

log() {
    printf '[SWEEP][%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf '[SWEEP][ERROR] %s\n' "$*" >&2
    exit 1
}

[[ -x "$RUN_SCRIPT" ]] ||
    die "Run script is not executable: $RUN_SCRIPT"

cd "$PROJECT_DIR"

TOTAL_RUNS=$(( \
    ${#NETWORKS[@]} * \
    ${#MAX_RATIOS[@]} * \
    ${#MPS_PERCENTAGES[@]} * \
    REPEATS \
))

CURRENT_RUN=0

log "Starting SplitMagic sweep"
log "Networks          : ${NETWORKS[*]}"
log "Max ratios       : ${MAX_RATIOS[*]}"
log "MPS percentages   : ${MPS_PERCENTAGES[*]}"
log "Repeats           : ${REPEATS}"
log "Total runs        : ${TOTAL_RUNS}"

for network in "${NETWORKS[@]}"; do
    for max_ratio in "${MAX_RATIOS[@]}"; do
        for mps in "${MPS_PERCENTAGES[@]}"; do
            for ((run_id = 0; run_id < REPEATS; run_id++)); do
                CURRENT_RUN=$((CURRENT_RUN + 1))

                log "============================================================"
                log "Run ${CURRENT_RUN}/${TOTAL_RUNS}"
                log "network=${network} Mbps"
                log "max_ratio=${max_ratio}"
                log "mps=${mps}%"
                log "run_id=${run_id}"
                log "============================================================"

                sudo -E \
                    PROJECT_DIR="$PROJECT_DIR" \
                    NETWORK_MBPS="$network" \
                    LATENCY_MS="$LATENCY_MS" \
                    DROP_RATIO="$max_ratio" \
                    SELECTION_POLICY="ratio" \
                    MPS_PERCENT="$mps" \
                    RUN_ID="$run_id" \
                    JIN_BATCH_SIZE="${BATCH_SIZE}" \
                    MAX_STEPS="$MAX_STEPS" \
                    MODEL="$MODEL" \
                    "$RUN_SCRIPT"

                log "Finished run ${CURRENT_RUN}/${TOTAL_RUNS}"

                sleep "$SLEEP_BETWEEN_RUNS"
            done
        done
    done
done

log "All experiments completed"
