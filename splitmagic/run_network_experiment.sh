#!/usr/bin/env bash
set -Eeuo pipefail

############################################################
# 설정
############################################################

# Node B로 통신할 때 사용하는 Node A의 네트워크 인터페이스
DEV="${DEV:-eth0}"

NETWORKS=(50 100 200 500 1000)
DROP_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# 각 조건 반복 횟수
RUNS="${RUNS:-1}"

# 실험 결과 최상위 디렉터리
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-experiment_results}"
RESULT_DIR="${RESULT_ROOT}/network_ratio_${TIMESTAMP}"

# 반드시 실제 Node A 실행 명령으로 수정
EXPERIMENT_CMD=(
    python3
    tests/test_node_a_resnet18.py
)

# tc TBF 설정
TBF_BURST="${TBF_BURST:-4mbit}"
TBF_LATENCY="${TBF_LATENCY:-400ms}"

CURRENT_RUN_DIR=""

mkdir -p "$RESULT_DIR"

############################################################
# 공통 함수
############################################################

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: $1" >&2
        exit 1
    fi
}

check_requirements() {
    require_command tc
    require_command ip
    require_command python3
    require_command stdbuf
    require_command tee

    if ! ip link show "$DEV" >/dev/null 2>&1; then
        echo "[ERROR] Network interface does not exist: $DEV" >&2
        echo
        echo "Available interfaces:"
        ip -brief link
        exit 1
    fi
}

############################################################
# 네트워크 설정
############################################################

clear_network_limit() {
    sudo tc qdisc del dev "$DEV" root 2>/dev/null || true
}

set_network_limit() {
    local network_mbps="$1"

    clear_network_limit

    sudo tc qdisc replace dev "$DEV" root tbf \
        rate "${network_mbps}mbit" \
        burst "$TBF_BURST" \
        latency "$TBF_LATENCY"

    local current_qdisc
    current_qdisc="$(tc qdisc show dev "$DEV")"

    if ! grep -qi "tbf" <<<"$current_qdisc"; then
        echo "[ERROR] Failed to apply network limit on $DEV" >&2
        echo "$current_qdisc" >&2
        return 1
    fi

    log "Applied network limit: ${network_mbps} Mbps"
    log "qdisc: ${current_qdisc}"
}

save_network_state() {
    local output_file="$1"

    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "interface=$DEV"
        echo

        echo "=== qdisc ==="
        tc -s qdisc show dev "$DEV"
        echo

        echo "=== interface ==="
        ip -s addr show dev "$DEV"
        echo

        echo "=== routing ==="
        ip route
    } >"$output_file" 2>&1
}

############################################################
# 정리
############################################################

cleanup() {
    local exit_code=$?

    set +e

    echo
    log "Restoring unlimited network on $DEV"

    clear_network_limit

    if [[ -n "$CURRENT_RUN_DIR" && -d "$CURRENT_RUN_DIR" ]]; then
        save_network_state \
            "$CURRENT_RUN_DIR/network_after_cleanup.txt" \
            2>/dev/null || true
    fi

    log "Cleanup completed"

    exit "$exit_code"
}

trap cleanup EXIT INT TERM

############################################################
# 전체 시스템 정보 저장
############################################################

save_system_metadata() {
    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "hostname=$(hostname)"
        echo "working_directory=$(pwd)"
        echo "interface=$DEV"
        echo "networks=${NETWORKS[*]}"
        echo "drop_ratios=${DROP_RATIOS[*]}"
        echo "runs=$RUNS"
        echo "experiment_command=${EXPERIMENT_CMD[*]}"
        echo

        echo "=== Python ==="
        python3 --version
        echo

        echo "=== interfaces ==="
        ip -brief addr
        echo

        echo "=== routing ==="
        ip route
        echo

        echo "=== initial qdisc ==="
        tc qdisc show dev "$DEV"
    } >"$RESULT_DIR/system_metadata.txt" 2>&1
}

############################################################
# 단일 실험
############################################################

run_one_experiment() {
    local network_mbps="$1"
    local drop_ratio="$2"
    local run_id="$3"

    local ratio_tag="${drop_ratio/./p}"

    CURRENT_RUN_DIR="${RESULT_DIR}/network_${network_mbps}mbps/ratio_${ratio_tag}/run_${run_id}"
    mkdir -p "$CURRENT_RUN_DIR"

    log "============================================================"
    log "network=${network_mbps} Mbps"
    log "drop_ratio=${drop_ratio}"
    log "run_id=${run_id}"
    log "output=${CURRENT_RUN_DIR}"
    log "============================================================"

    set_network_limit "$network_mbps"

    save_network_state "$CURRENT_RUN_DIR/network_before.txt"

    local start_ns
    local end_ns
    local exit_code

    start_ns="$(date +%s%N)"

    set +e

    DROP_RATIO="$drop_ratio" \
    NETWORK_MBPS="$network_mbps" \
    NETWORK_INTERFACE="$DEV" \
    RUN_ID="$run_id" \
    RESULT_CSV="$CURRENT_RUN_DIR/result.csv" \
    EXPERIMENT_OUTPUT_DIR="$CURRENT_RUN_DIR" \
        stdbuf -oL -eL "${EXPERIMENT_CMD[@]}" \
        > >(tee "$CURRENT_RUN_DIR/node_a_stdout.log") \
        2> >(tee "$CURRENT_RUN_DIR/node_a_stderr.log" >&2)

    exit_code=$?

    set -e

    end_ns="$(date +%s%N)"

    save_network_state "$CURRENT_RUN_DIR/network_after.txt"

    python3 - \
        "$CURRENT_RUN_DIR/run_metadata.csv" \
        "$network_mbps" \
        "$drop_ratio" \
        "$run_id" \
        "$start_ns" \
        "$end_ns" \
        "$exit_code" \
        "$DEV" <<'PY'
import csv
import os
import sys
from datetime import datetime, timezone

(
    output_path,
    network_mbps,
    drop_ratio,
    run_id,
    start_ns,
    end_ns,
    exit_code,
    interface,
) = sys.argv[1:]

start_ns = int(start_ns)
end_ns = int(end_ns)

row = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "hostname": os.uname().nodename,
    "network_mbps": int(network_mbps),
    "network_interface": interface,
    "drop_ratio": float(drop_ratio),
    "run_id": int(run_id),
    "start_epoch_ns": start_ns,
    "end_epoch_ns": end_ns,
    "shell_elapsed_ms": (end_ns - start_ns) / 1_000_000,
    "exit_code": int(exit_code),
}

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
PY

    if [[ "$exit_code" -ne 0 ]]; then
        log "[ERROR] Experiment failed: exit_code=${exit_code}"
        return "$exit_code"
    fi

    log "Experiment completed"
}

############################################################
# 메인
############################################################

main() {
    check_requirements
    save_system_metadata

    log "Results: $RESULT_DIR"
    log "Interface: $DEV"

    echo
    log "Initial qdisc:"
    tc qdisc show dev "$DEV"

    for network_mbps in "${NETWORKS[@]}"; do
        for drop_ratio in "${DROP_RATIOS[@]}"; do
            for ((run_id = 0; run_id < RUNS; run_id++)); do

                run_one_experiment \
                    "$network_mbps" \
                    "$drop_ratio" \
                    "$run_id"

                # GPU와 시스템이 다음 실험 전에 안정화될 시간
                sleep 2
            done
        done
    done

    CURRENT_RUN_DIR=""

    log "All experiments completed"
    log "Results saved in: $RESULT_DIR"
}

main "$@"
