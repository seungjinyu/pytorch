#!/usr/bin/env bash

set -u

if [[ "$EUID" -eq 0 ]]; then
    echo "[ERROR] Do not run this script with sudo."
    echo "[ERROR] Run it as a normal user: ./run_all_nsys.sh"
    exit 1
fi

PERCENTAGES=(100 50 25 10)
NETWORK_MBPS_LIST=(1000 500 200 100 50)
DROP_RATIOS=(0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

NODE_B_SCRIPT="tests/test_node_b_resnet18.py"
OUTPUT_DIR="./nsys_results/$TIMESTAMP"
IMPORTER="/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"


# Node A SSH 정보
NODE_A_HOST="ubuntu@pi"
NODE_A_PROJECT_DIR="/home/ubuntu/splitmagic_project"
NODE_A_SCRIPT="tests/test_node_a_resnet18.py"
NODE_A_PYTHON="/home/ubuntu/venv/splitmagic/bin/python"

NODE_A_IFACE="eth0"
NODE_B_IFACE="enp3s0"

CPU_THREADS=1
STARTUP_WAIT_SEC=5

NODE_B_PID=""

mkdir -p "$OUTPUT_DIR"

apply_network_limit() {
    local mbps="$1"

    echo
    echo "[NETWORK] Applying ${mbps} Mbps"

    echo "[NODE B] applying limit"
    sudo -n tc qdisc replace dev "$NODE_B_IFACE" root tbf \
        rate "${mbps}mbit" \
        burst 1mb \
        latency 100ms

    echo "[NODE A] applying limit"
    ssh "$NODE_A_HOST" "
        sudo -n tc qdisc replace dev '$NODE_A_IFACE' root tbf \
            rate '${mbps}mbit' \
            burst 1mb \
            latency 100ms
    "

    echo "[NODE B qdisc]"
    tc qdisc show dev "$NODE_B_IFACE"

    echo "[NODE A qdisc]"
    ssh "$NODE_A_HOST" \
        "tc qdisc show dev '$NODE_A_IFACE'"

    # 실제로 둘 다 TBF인지 확인
    if ! tc qdisc show dev "$NODE_B_IFACE" | grep -q "qdisc tbf"; then
        echo "[ERROR] Node B network limit was not applied"
        return 1
    fi

    if ! ssh "$NODE_A_HOST" \
        "tc qdisc show dev '$NODE_A_IFACE'" | grep -q "qdisc tbf"; then
        echo "[ERROR] Node A network limit was not applied"
        return 1
    fi
}


clear_network_limit() {
    echo "[NETWORK] Clearing limits"

    sudo -n tc qdisc del dev "$NODE_B_IFACE" root \
        2>/dev/null || true

    ssh "$NODE_A_HOST" "
        sudo -n tc qdisc del dev '$NODE_A_IFACE' root \
            2>/dev/null || true
    " || true
}


cleanup() {
    if [[ -n "${NODE_B_PID:-}" ]] &&
       kill -0 "$NODE_B_PID" 2>/dev/null; then

        echo "[CLEANUP] stopping Node B PID=$NODE_B_PID"
        kill -INT "$NODE_B_PID" 2>/dev/null || true
        sleep 2
        kill -TERM "$NODE_B_PID" 2>/dev/null || true
        wait "$NODE_B_PID" 2>/dev/null || true
    fi

    clear_network_limit
}


trap cleanup EXIT INT TERM


for JIN_NETWORK_MBPS in "${NETWORK_MBPS_LIST[@]}"; do

    # 이 network 값은 아래 모든 MPS/ratio 실험에 적용됨
    apply_network_limit "$JIN_NETWORK_MBPS"

    for JIN_AUTO_DROP_RATIO in "${DROP_RATIOS[@]}"; do
        for PERCENT in "${PERCENTAGES[@]}"; do

            NAME="splitmagic_${TIMESTAMP}_net${JIN_NETWORK_MBPS}_drop${JIN_AUTO_DROP_RATIO}_mps${PERCENT}_threads${CPU_THREADS}"
            BASE="${OUTPUT_DIR}/${NAME}"

            echo
            echo "=================================================="
            echo "[START]"
            echo "MPS=${PERCENT}%"
            echo "NETWORK=${JIN_NETWORK_MBPS} Mbps"
            echo "DROP_RATIO=${JIN_AUTO_DROP_RATIO}"
            echo "=================================================="

            rm -f \
                "${BASE}.qdstrm" \
                "${BASE}.nsys-rep" \
                "${BASE}.sqlite"

            OMP_NUM_THREADS="$CPU_THREADS" \
            MKL_NUM_THREADS="$CPU_THREADS" \
            OPENBLAS_NUM_THREADS="$CPU_THREADS" \
            NUMEXPR_NUM_THREADS="$CPU_THREADS" \
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$PERCENT" \
            JIN_NETWORK_MBPS="$JIN_NETWORK_MBPS" \
            JIN_AUTO_DROP_RATIO="$JIN_AUTO_DROP_RATIO" \
            nsys profile \
                --force-overwrite=true \
                --trace=cuda,nvtx,osrt \
                --sample=none \
                --cuda-memory-usage=true \
                -o "$BASE" \
                python3 "$NODE_B_SCRIPT" \
                > "${BASE}_node_b.log" 2>&1 &

            NODE_B_PID=$!

            echo "[NODE B] started PID=$NODE_B_PID"
            echo "[WAIT] waiting ${STARTUP_WAIT_SEC}s"

            sleep "$STARTUP_WAIT_SEC"

            if ! kill -0 "$NODE_B_PID" 2>/dev/null; then
                echo "[ERROR] Node B exited before Node A started"
                tail -n 50 "${BASE}_node_b.log"
                exit 1
            fi

            echo "[NODE A] starting through SSH"

            ssh "$NODE_A_HOST" "
                set -e
                cd '$NODE_A_PROJECT_DIR'

                JIN_NETWORK_MBPS='$JIN_NETWORK_MBPS' \
                JIN_AUTO_DROP_RATIO='$JIN_AUTO_DROP_RATIO' \
                OMP_NUM_THREADS='$CPU_THREADS' \
                MKL_NUM_THREADS='$CPU_THREADS' \
                OPENBLAS_NUM_THREADS='$CPU_THREADS' \
                NUMEXPR_NUM_THREADS='$CPU_THREADS' \
                '$NODE_A_PYTHON' '$NODE_A_SCRIPT'
            " > "${BASE}_node_a.log" 2>&1

            NODE_A_STATUS=$?

            if [[ "$NODE_A_STATUS" -ne 0 ]]; then
                echo "[ERROR] Node A failed: status=$NODE_A_STATUS"
                tail -n 50 "${BASE}_node_a.log"
                exit "$NODE_A_STATUS"
            fi

            echo "[NODE A] finished"
            echo "[NODE B] stopping profiler"

            sleep 3

            kill -INT "$NODE_B_PID" 2>/dev/null || true

            for _ in $(seq 1 30); do
                if ! kill -0 "$NODE_B_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done

            if kill -0 "$NODE_B_PID" 2>/dev/null; then
                echo "[WARN] sending SIGTERM"
                kill -TERM "$NODE_B_PID" 2>/dev/null || true
                sleep 2
            fi

            wait "$NODE_B_PID" 2>/dev/null || true
            NODE_B_PID=""

            if [[ -f "${BASE}.qdstrm" ]]; then
                echo "[CONVERT] qdstrm -> nsys-rep"

                "$IMPORTER" \
                    "${BASE}.qdstrm" \
                    -o "${BASE}.nsys-rep"

            elif [[ -f "${BASE}.nsys-rep" ]]; then
                echo "[INFO] nsys-rep already exists"

            else
                echo "[ERROR] no profiling output found"
                exit 1
            fi

            echo "[STATS] generating reports"

            nsys stats \
                --report nvtxsum \
                "${BASE}.nsys-rep" \
                > "${BASE}_nvtxsum.txt"

            nsys stats \
                --report cudaapisum \
                "${BASE}.nsys-rep" \
                > "${BASE}_cudaapisum.txt"

            nsys stats \
                --report gpukernsum \
                "${BASE}.nsys-rep" \
                > "${BASE}_gpukernsum.txt"

            nsys stats \
                --report gpumemtimesum \
                "${BASE}.nsys-rep" \
                > "${BASE}_gpumemtimesum.txt"

            echo "[DONE]"
            echo "MPS=${PERCENT}%"
            echo "NETWORK=${JIN_NETWORK_MBPS} Mbps"
            echo "DROP_RATIO=${JIN_AUTO_DROP_RATIO}"

        done
    done
done

echo
echo "[ALL DONE]"