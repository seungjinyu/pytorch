#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

MERGED_ROOT="${1:-${PROJECT_ROOT}/merged_profiles}"
OUTPUT_ROOT="${2:-${PROJECT_ROOT}/menus_offline}"
NSYS_ROOT="${3:-${PROJECT_ROOT}/nsys_results}"

PYTHON_BIN="${PYTHON_BIN:-/home/syu23/miniconda3/envs/torch-build/bin/python3}"
FORCE="${FORCE:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

BUILDER="${PROJECT_ROOT}/tools/build_menu_offline.py"

cd "${PROJECT_ROOT}"

if [[ ! -f "${BUILDER}" ]]; then
    echo "[ERROR] builder not found: ${BUILDER}" >&2
    exit 1
fi

if [[ ! -d "${MERGED_ROOT}" ]]; then
    echo "[ERROR] merged root not found: ${MERGED_ROOT}" >&2
    exit 1
fi

PLAN="$(
    find "${NSYS_ROOT}" \
        -type f \
        -path '*resnet18*' \
        -name 'jin_template_plan_a.tsv' \
    | sort \
    | tail -1
)"

if [[ -z "${PLAN}" ]]; then
    echo "[ERROR] ResNet18 template plan not found" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] merged_root=${MERGED_ROOT}"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] plan=${PLAN}"

success=0
failed=0
skipped=0

while IFS= read -r -d '' recompute_profile; do
    filename="$(basename "${recompute_profile}")"
    condition="${filename%_recompute_profile.csv}"

    inject_profile="${recompute_profile%_recompute_profile.csv}_inject_profile.csv"
    output_profile="${recompute_profile%_recompute_profile.csv}_output_to_cpu_profile.csv"

    output_dir="${OUTPUT_ROOT}/${condition}"
    final_menu="${output_dir}/final_menu.csv"

    echo
    echo "============================================================"
    echo "[CONDITION] ${condition}"
    echo "[RECOMPUTE] ${recompute_profile}"
    echo "[INJECT] ${inject_profile}"
    echo "[OUTPUT_TO_CPU] ${output_profile}"
    echo "============================================================"

    if [[ ! -f "${inject_profile}" ]]; then
        echo "[FAIL] inject profile not found"
        failed=$((failed + 1))
        continue
    fi

    if [[ ! -f "${output_profile}" ]]; then
        echo "[FAIL] output-to-CPU profile not found: ${output_profile}"
        failed=$((failed + 1))
        continue
    fi

    if [[ -s "${final_menu}" && "${FORCE}" != "1" ]]; then
        echo "[SKIP] existing: ${final_menu}"
        skipped=$((skipped + 1))
        continue
    fi

    if "${PYTHON_BIN}" "${BUILDER}" \
        --model resnet18 \
        --plan "${PLAN}" \
        --recompute-profile "${recompute_profile}" \
        --inject-profile "${inject_profile}" \
        --output-profile "${output_profile}" \
        --output-dir "${output_dir}" \
        --batch-size "${BATCH_SIZE}"\
        --network-mbps "${NETWORK_MBPS:-1000}" \
        --mps-percent "${condition##*mps}" \
        --profile-steps "${PROFILE_STEPS:-1000}" \
        --metric "${METRIC:-median}" \
        --drop-ratio "-0.99"; then

        success=$((success + 1))
    else
        failed=$((failed + 1))
    fi

done < <(
    find "${MERGED_ROOT}" \
        -maxdepth 1 \
        -type f \
        -name 'resnet18_*_recompute_profile.csv' \
        -print0 |
    sort -z
)

echo
echo "================ SUMMARY ================"
echo "success=${success}"
echo "skipped=${skipped}"
echo "failed=${failed}"
echo "output=${OUTPUT_ROOT}"

if [[ "${failed}" -gt 0 ]]; then
    exit 2
fi