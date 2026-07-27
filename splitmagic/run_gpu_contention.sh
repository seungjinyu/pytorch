#!/usr/bin/env bash
set -Eeuo pipefail

############################################################
# 설정
############################################################

GPU_CONTENTION_SCRIPT="${GPU_CONTENTION_SCRIPT:-gpu_contention_128.py}"

GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-0.5}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-gpu_contention_results}"
RESULT_DIR="${RESULT_ROOT}/contention_${TIMESTAMP}"

GPU_CONTENTION_PID=""
GPU_MONITOR_PID=""

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
    require_command python3
    require_command nvidia-smi
    require_command stdbuf

    if [[ ! -f "$GPU_CONTENTION_SCRIPT" ]]; then
        echo "[ERROR] Cannot find GPU contention script:" >&2
        echo "        $GPU_CONTENTION_SCRIPT" >&2
        exit 1
    fi
}

############################################################
# GPU 상태 저장
############################################################

save_gpu_snapshot() {
    local output_file="$1"

    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo

        nvidia-smi
        echo

        echo "=== GPU query ==="

        nvidia-smi \
            --query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem \
            --format=csv
        echo

        echo "=== compute processes ==="

        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv 2>/dev/null || true
    } >"$output_file" 2>&1
}

############################################################
# GPU monitor
############################################################

start_gpu_monitor() {
    local output_file="$RESULT_DIR/gpu_samples.csv"

    log "Starting GPU monitor"

    (
        echo "timestamp,index,name,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,temperature_c,power_draw_w,power_limit_w,clocks_sm_mhz,clocks_memory_mhz"

        while true; do
            timestamp="$(date --iso-8601=ns)"

            nvidia-smi \
                --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem \
                --format=csv,noheader,nounits |
            while IFS= read -r gpu_line; do
                echo "${timestamp},${gpu_line}"
            done

            sleep "$GPU_SAMPLE_INTERVAL"
        done
    ) >>"$output_file" 2>&1 &

    GPU_MONITOR_PID=$!

    log "GPU monitor PID: $GPU_MONITOR_PID"
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID:-}" ]] &&
       kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then

        log "Stopping GPU monitor PID=$GPU_MONITOR_PID"

        kill -TERM "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi

    GPU_MONITOR_PID=""
}

############################################################
# GPU contention
############################################################

start_gpu_contention() {
    log "Starting GPU contention"
    log "Script: $GPU_CONTENTION_SCRIPT"

    # gpu_contention.py가 인자를 요구하면 아래 줄 뒤에 인자를 추가
    stdbuf -oL -eL \
        python3 "$GPU_CONTENTION_SCRIPT" \
        >"$RESULT_DIR/gpu_contention_stdout.log" \
        2>"$RESULT_DIR/gpu_contention_stderr.log" &

    GPU_CONTENTION_PID=$!

    sleep 2

    if ! kill -0 "$GPU_CONTENTION_PID" 2>/dev/null; then
        echo "[ERROR] gpu_contention.py terminated immediately." >&2

        echo
        echo "=== stdout ==="
        cat "$RESULT_DIR/gpu_contention_stdout.log" 2>/dev/null || true

        echo
        echo "=== stderr ==="
        cat "$RESULT_DIR/gpu_contention_stderr.log" 2>/dev/null || true

        return 1
    fi

    log "GPU contention PID: $GPU_CONTENTION_PID"
}

stop_gpu_contention() {
    if [[ -n "${GPU_CONTENTION_PID:-}" ]] &&
       kill -0 "$GPU_CONTENTION_PID" 2>/dev/null; then

        log "Stopping GPU contention PID=$GPU_CONTENTION_PID"

        kill -TERM "$GPU_CONTENTION_PID" 2>/dev/null || true

        for _ in {1..30}; do
            if ! kill -0 "$GPU_CONTENTION_PID" 2>/dev/null; then
                break
            fi

            sleep 0.1
        done

        if kill -0 "$GPU_CONTENTION_PID" 2>/dev/null; then
            log "Contention process did not stop; sending SIGKILL"
            kill -KILL "$GPU_CONTENTION_PID" 2>/dev/null || true
        fi

        wait "$GPU_CONTENTION_PID" 2>/dev/null || true
    fi

    GPU_CONTENTION_PID=""
}

############################################################
# 종료 처리
############################################################

cleanup() {
    local exit_code=$?

    set +e

    echo
    log "Cleanup started"

    save_gpu_snapshot "$RESULT_DIR/gpu_before_cleanup.txt" \
        2>/dev/null || true

    stop_gpu_contention
    stop_gpu_monitor

    save_gpu_snapshot "$RESULT_DIR/gpu_after_cleanup.txt" \
        2>/dev/null || true

    log "Cleanup completed"
    log "Logs saved in: $RESULT_DIR"

    exit "$exit_code"
}

trap cleanup EXIT INT TERM

############################################################
# 시스템 정보
############################################################

save_system_metadata() {
    {
        echo "timestamp=$(date --iso-8601=seconds)"
        echo "hostname=$(hostname)"
        echo "working_directory=$(pwd)"
        echo "gpu_contention_script=$GPU_CONTENTION_SCRIPT"
        echo "gpu_sample_interval=$GPU_SAMPLE_INTERVAL"
        echo

        echo "=== Python ==="
        python3 --version
        echo

        echo "=== PyTorch ==="

        python3 - <<'PY'
try:
    import torch

    print("torch_version:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", torch.version.cuda)

    if torch.cuda.is_available():
        print("gpu_count:", torch.cuda.device_count())

        for index in range(torch.cuda.device_count()):
            print(
                f"gpu_{index}:",
                torch.cuda.get_device_name(index),
            )
except Exception as exc:
    print("PyTorch information unavailable:", repr(exc))
PY

        echo
        echo "=== NVIDIA SMI ==="
        nvidia-smi
    } >"$RESULT_DIR/system_metadata.txt" 2>&1
}

############################################################
# 메인
############################################################

main() {
    check_requirements
    save_system_metadata
    save_gpu_snapshot "$RESULT_DIR/gpu_before.txt"

    log "Result directory: $RESULT_DIR"

    start_gpu_monitor
    start_gpu_contention

    sleep 3

    save_gpu_snapshot "$RESULT_DIR/gpu_after_contention_start.txt"

    echo
    log "GPU contention is running."
    log "Now start run_network_experiment.sh on Node A."
    log "Press Ctrl+C after all Node A experiments finish."
    echo

    while true; do
        if ! kill -0 "$GPU_CONTENTION_PID" 2>/dev/null; then
            echo "[ERROR] gpu_contention.py unexpectedly terminated." >&2
            return 1
        fi

        if ! kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
            echo "[ERROR] GPU monitor unexpectedly terminated." >&2
            return 1
        fi

        sleep 2
    done
}

main "$@"
