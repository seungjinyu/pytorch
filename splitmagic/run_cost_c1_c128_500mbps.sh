#!/bin/bash

set -euo pipefail

############################
# configuration
############################

NODE_B_HOST="syu23@oryx"
NODE_B_IP="10.32.137.57"

NETWORKS=(500)
CONCURRENCIES=(1 128)
REPEATS=5

RESULT_CSV="./cost_c1_c128_500mbps.csv"
NETWORK_CSV="./cost_c1_c128_500mbps_tc.csv"

SELECTION_POLICY="cost"
MIN_BENEFIT_MS="0.0"
MAX_COST_DROP_RATIO="0.99"

INTERFACE="eth0"
NODE_A_HOSTNAME="$(hostname)"
RUN_ID=1

declare -A RECOMPUTE_COST_CSVS=(
    [1]="$PWD/recompute_menu_resnet18_bs32_c1.csv"
    [128]="$PWD/recompute_menu_resnet18_bs32_c128.csv"
)

declare -A NODE_B_PROFILE_PATHS=(
    [1]="/home/syu23/seungjin/pytorch/splitmagic/profiling/recompute_node_profile_c1.csv"
    [128]="/home/syu23/seungjin/pytorch/splitmagic/profiling/recompute_node_profile_c128.csv"
)

############################
# output initialization
############################

rm -f "${RESULT_CSV}"
rm -f "${NETWORK_CSV}"

echo \
"run_id,repeat,hostname,selection_policy,concurrency,network_interface,tc_enabled,network_limit_mbps,tc_bytes_before,tc_bytes_after,tc_bytes_delta" \
> "${NETWORK_CSV}"

############################
# tc helpers
############################

clear_tc() {
    sudo tc qdisc del dev "${INTERFACE}" root 2>/dev/null || true
}

set_tc_limit() {
    local limit="$1"

    clear_tc

    if [[ "${limit}" == "0" ]]; then
        echo "[NETWORK] unlimited"
        return
    fi

    echo "[NETWORK] limit=${limit}mbit"

    sudo tc qdisc replace dev "${INTERFACE}" root handle 1: htb default 20
    sudo tc class add dev "${INTERFACE}" parent 1: classid 1:1 htb rate 1gbit ceil 1gbit
    sudo tc class add dev "${INTERFACE}" parent 1:1 classid 1:10 htb rate "${limit}mbit" ceil "${limit}mbit"
    sudo tc class add dev "${INTERFACE}" parent 1:1 classid 1:20 htb rate 1gbit ceil 1gbit
    sudo tc filter add dev "${INTERFACE}" protocol ip parent 1: prio 1 u32 \
        match ip dst "${NODE_B_IP}/32" flowid 1:10
}

get_tc_bytes() {
    sudo tc -s class show dev "${INTERFACE}" |
        awk '
            /class htb 1:10/ { found=1; next }
            found && /Sent/ { print $2; exit }
        '
}

############################
# node b control
############################

stop_node_b() {
    echo "[NODE B] stopping old process..."

    ssh "${NODE_B_HOST}" '
        pkill -TERM -f "[t]ests/test_node_b_resnet18.py" || true

        for i in $(seq 1 15); do
            process_alive=0
            port_busy=0

            pgrep -f "[t]ests/test_node_b_resnet18.py" >/dev/null && process_alive=1
            ss -ltnH 2>/dev/null | awk "{print \$4}" | grep -Eq "(^|:|\])5556$" && port_busy=1

            if [ "$process_alive" -eq 0 ] && [ "$port_busy" -eq 0 ]; then
                echo "[NODE B] process stopped and port 5556 is free"
                exit 0
            fi

            echo "[NODE B] waiting for shutdown process_alive=$process_alive port_busy=$port_busy"
            sleep 1
        done

        echo "[NODE B] graceful shutdown timed out; forcing cleanup"
        pkill -KILL -f "[t]ests/test_node_b_resnet18.py" || true
        fuser -k 5556/tcp 2>/dev/null || true
        sleep 2

        if pgrep -f "[t]ests/test_node_b_resnet18.py" >/dev/null; then
            echo "[ERROR] Node B process still exists"
            pgrep -af "[t]ests/test_node_b_resnet18.py" || true
            exit 1
        fi

        if ss -ltnH 2>/dev/null | awk "{print \$4}" | grep -Eq "(^|:|\])5556$"; then
            echo "[ERROR] port 5556 is still occupied"
            ss -ltnp | grep ":5556" || true
            exit 1
        fi

        echo "[NODE B] forced cleanup completed"
    '
}

start_node_b() {
    local concurrency="$1"
    local profile_path="$2"

    stop_node_b

    echo "[NODE B] checking startup conditions..."
    echo "[NODE B] concurrency=${concurrency}"
    echo "[NODE B] profile_path=${profile_path}"

    ssh "${NODE_B_HOST}" bash -s -- "${profile_path}" <<'REMOTE_CHECK'
set -euo pipefail
profile_path="$1"

if [[ ! -f "${profile_path}" ]]; then
    echo "[ERROR] Node B profile not found: ${profile_path}"
    exit 1
fi

if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:|\])5556$'; then
    echo "[ERROR] port 5556 occupied before Node B startup"
    ss -ltnp | grep ':5556' || true
    exit 1
fi

 echo "[NODE B] profile exists: ${profile_path}"
 echo "[NODE B] port 5556 is free"
REMOTE_CHECK

    echo "[NODE B] starting..."

    ssh "${NODE_B_HOST}" bash -s -- "${concurrency}" "${profile_path}" <<'REMOTE_START'
set -euo pipefail

concurrency="$1"
profile_path="$2"

cd /home/syu23/seungjin/pytorch/splitmagic
rm -f /tmp/node_b_resnet18.log

nohup env \
    PYTHONPATH="/home/syu23/seungjin/pytorch:/home/syu23/seungjin/pytorch/splitmagic:/home/syu23/torchvision-0.17:${PYTHONPATH:-}" \
    JIN_RECOMPUTE_PROFILE_PATH="${profile_path}" \
    JIN_RECOMPUTE_PROFILE_CONCURRENCY="${concurrency}" \
    /home/syu23/miniconda3/envs/torch-build/bin/python3 -u \
    tests/test_node_b_resnet18.py \
    > /tmp/node_b_resnet18.log 2>&1 \
    < /dev/null &

 echo "[NODE B START REQUESTED]"
 echo "pid=$!"
 echo "JIN_RECOMPUTE_PROFILE_PATH=${profile_path}"
 echo "JIN_RECOMPUTE_PROFILE_CONCURRENCY=${concurrency}"
REMOTE_START

    echo "[NODE B] waiting for listening..."

    local ready=0

    for _ in $(seq 1 60); do
        if ssh "${NODE_B_HOST}" "grep -q '\[Node B\] listening' /tmp/node_b_resnet18.log 2>/dev/null"; then
            ready=1
            break
        fi

        if ! ssh "${NODE_B_HOST}" "pgrep -f '[t]ests/test_node_b_resnet18.py' >/dev/null"; then
            echo "[ERROR] Node B exited during startup"
            ssh "${NODE_B_HOST}" "tail -100 /tmp/node_b_resnet18.log" || true
            return 1
        fi

        sleep 1
    done

    if [[ "${ready}" -ne 1 ]]; then
        echo "[ERROR] Node B did not become ready"
        ssh "${NODE_B_HOST}" "tail -100 /tmp/node_b_resnet18.log" || true
        return 1
    fi

    echo "[NODE B] ready"
}

############################
# cleanup on exit
############################

cleanup() {
    stop_node_b || true
    clear_tc
}

trap cleanup EXIT

############################
# preflight checks
############################

for concurrency in "${CONCURRENCIES[@]}"; do
    recompute_cost_csv="${RECOMPUTE_COST_CSVS[$concurrency]}"

    if [[ ! -f "${recompute_cost_csv}" ]]; then
        echo "[ERROR] Node A recompute menu not found: ${recompute_cost_csv}"
        exit 1
    fi
done

############################
# experiment loop
############################

for concurrency in "${CONCURRENCIES[@]}"; do
    recompute_cost_csv="${RECOMPUTE_COST_CSVS[$concurrency]}"
    node_b_profile_path="${NODE_B_PROFILE_PATHS[$concurrency]}"

    echo
    echo "####################################"
    echo "concurrency=${concurrency}"
    echo "Node A menu=${recompute_cost_csv}"
    echo "Node B profile=${node_b_profile_path}"
    echo "####################################"

    for network in "${NETWORKS[@]}"; do
        set_tc_limit "${network}"

        echo "[TC CONFIG]"
        sudo tc -s class show dev "${INTERFACE}" | grep -A2 "class htb 1:10" || true

        for repeat in $(seq 1 "${REPEATS}"); do
            echo
            echo "===================================="
            echo "policy=${SELECTION_POLICY}"
            echo "concurrency=${concurrency}"
            echo "network=${network}"
            echo "repeat=${repeat}/${REPEATS}"
            echo "run_id=${RUN_ID}"
            echo "===================================="

            start_node_b "${concurrency}" "${node_b_profile_path}"

            TC_BYTES_BEFORE="$(get_tc_bytes)"
            TC_BYTES_BEFORE="${TC_BYTES_BEFORE:-0}"

            JIN_SELECTION_POLICY="${SELECTION_POLICY}" \
            JIN_RECOMPUTE_COST_CSV="${recompute_cost_csv}" \
            JIN_RECOMPUTE_PROFILE_CONCURRENCY="${concurrency}" \
            JIN_NETWORK_MBPS="${network}" \
            JIN_AUTO_DROP_RATIO="0.0" \
            JIN_MIN_BENEFIT_MS="${MIN_BENEFIT_MS}" \
            JIN_MAX_COST_DROP_RATIO="${MAX_COST_DROP_RATIO}" \
            JIN_EXPERIMENT_RUN_ID="${RUN_ID}" \
            JIN_MAX_STEPS="1" \
            JIN_EXPERIMENT_CSV="${RESULT_CSV}" \
            JIN_NODE_A_CSV="./node_a_cost_c${concurrency}_${network}mbps_run${RUN_ID}.csv" \
            python3 -u tests/test_node_a_cost_resnet18.py

            TC_BYTES_AFTER="$(get_tc_bytes)"
            TC_BYTES_AFTER="${TC_BYTES_AFTER:-0}"
            TC_BYTES_DELTA=$((TC_BYTES_AFTER - TC_BYTES_BEFORE))

            echo \
            "${RUN_ID},${repeat},${NODE_A_HOSTNAME},${SELECTION_POLICY},${concurrency},${INTERFACE},1,${network},${TC_BYTES_BEFORE},${TC_BYTES_AFTER},${TC_BYTES_DELTA}" \
            >> "${NETWORK_CSV}"

            echo "[NETWORK_STATS] run_id=${RUN_ID} concurrency=${concurrency} network=${network} before=${TC_BYTES_BEFORE} after=${TC_BYTES_AFTER} delta=${TC_BYTES_DELTA}"

            stop_node_b
            RUN_ID=$((RUN_ID + 1))
            sleep 2
        done
    done
done

echo
echo "[ALL DONE]"
echo "[RESULT CSV]  ${RESULT_CSV}"
echo "[NETWORK CSV] ${NETWORK_CSV}"
