#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

NSYS_ROOT="${1:-${PROJECT_ROOT}/nsys_results}"

MENU_RATIO="0.99"

cd "${PROJECT_ROOT}"

echo "[MENU_PROFILE] root=${NSYS_ROOT}"
echo "[MENU_PROFILE] required_ratio=${MENU_RATIO}"

FORCE=1 \
PROFILE_RATIO="${MENU_RATIO}" \
./make_all_nvtx_profiles.sh "${NSYS_ROOT}"

FORCE=1 \
PROFILE_RATIO="${MENU_RATIO}" \
./merge_all_nvtx_profiles.sh \
    "${NSYS_ROOT}" \
    "${PROJECT_ROOT}/merged_profiles"