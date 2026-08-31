set -Eeuo pipefail

# ============================================================
# SplitMagic offline menu pipeline
# ============================================================

PROJECT_ROOT="$PWD"
NSYS_ROOT="${PROJECT_ROOT}/nsys_results"
MERGED_ROOT="${PROJECT_ROOT}/merged_profiles"
MENU_ROOT="${PROJECT_ROOT}/menus_offline"

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
MODEL="${MODEL:-resnet18_imagenet}"
BATCH_SIZE="${BATCH_SIZE:-32}"

NETWORK_MBPS="${NETWORK_MBPS:-1000}"
PROFILE_RATIO="${PROFILE_RATIO:-1.0}"

MAX_STEPS="${MAX_STEPS:-1000}"
REPEATS="${REPEATS:-1}"

METRIC="${METRIC:-median}"
FORCE="${FORCE:-1}"

echo "============================================================"
echo "[1/6] Run profiling experiments"
echo "============================================================"

MODEL="${MODEL}" \
MAX_STEPS="${MAX_STEPS}" \
REPEATS="${REPEATS}" \
JIN_BATCH_SIZE="${BATCH_SIZE}" \
./run_all_local_nsys.sh

echo
echo "============================================================"
echo "[2/6] Convert qdstrm to nsys-rep"
echo "============================================================"

FORCE=${FORCE} \
./convert_all_qdstrm.sh

echo
echo "============================================================"
echo "[3/6] Export Nsight CSV reports"
echo "============================================================"

FORCE=${FORCE} \
./export_nsys_csv.sh

echo
echo "============================================================"
echo "[4/6] Extract ratio=1.0 NVTX profiles"
echo "============================================================"
// PROFILE_RATIO=1.0 METRIC=median FORCE=1 ./make_all_nvtx_profiles.sh     "${NSYS_ROOT}"

PROFILE_RATIO="${PROFILE_RATIO}" \
METRIC="${METRIC}" \
FORCE="${FORCE}" \
./make_all_nvtx_profiles.sh \
    "${NSYS_ROOT}"

echo
echo "============================================================"
echo "[5/6] Merge ratio=${PROFILE_RATIO} profiles by MPS condition"
echo "============================================================"


// PROFILE_RATIO=1.0 FORCE=1 ./merge_all_nvtx_profiles.sh     "${NSYS_ROOT}"     "${MERGED_ROOT}"
PROFILE_RATIO="${PROFILE_RATIO}" \
FORCE="${FORCE}" \
./merge_all_nvtx_profiles.sh \
    "${NSYS_ROOT}" \
    "${MERGED_ROOT}"

echo
echo "============================================================"
echo "[6/6] Build final offline menus"
echo "============================================================"

<!-- MODEL=resnet18_imagenet \
BATCH_SIZE=32 \
FORCE=1 \
tools/build_all_menus_offline.sh -->
<!-- MODEL=resnet18 BATCH_SIZE=32 FORCE=1 tools/build_all_menus_offline.sh -->

// tools/build_all_menus_offline.sh
FORCE="${FORCE}" \
NETWORK_MBPS="${NETWORK_MBPS}" \
PROFILE_STEPS="${MAX_STEPS}" \
METRIC="${METRIC}" \
MODEL="${MODEL}" \
BATCH_SIZE="${BATCH_SIZE}" \
./tools/build_all_menus_offline.sh \
    "${MERGED_ROOT}" \
    "${MENU_ROOT}" \
    "${NSYS_ROOT}"

echo
echo "============================================================"
echo "[VERIFY] Generated menus and metadata"
echo "============================================================"

find "${MENU_ROOT}" \
    -maxdepth 2 \
    -type f \
    \( \
        -name 'final_menu.csv' \
        -o -name 'metadata.json' \
    \) \
    | sort

echo
echo "============================================================"
echo "[VERIFY] Menu coverage"
echo "============================================================"

MODEL="${MODEL}" python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd


model = os.environ["MODEL"]

paths = sorted(
    Path("./menus_offline").glob(
        f"{model}_cuda_mps*/final_menu.csv"
    )
)

if not paths:
    raise RuntimeError(
        f"No final_menu.csv files were generated "
        f"for model={model}"
    )

for path in paths:
    df = pd.read_csv(path)

    required = {
        "recompute_ms",
        "output_to_cpu_ms",
        "patch_ms",
        "output_profile_missing",
        "inject_profile_missing",
    }

    missing_columns = required - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            f"{path}: missing columns "
            f"{sorted(missing_columns)}"
        )

    usable = (
        df["recompute_ms"].notna()
        & df["output_to_cpu_ms"].notna()
        & df["patch_ms"].notna()
        & df["output_profile_missing"].eq(0)
        & df["inject_profile_missing"].eq(0)
    )

    print(
        f"{path.parent.name}: "
        f"total={len(df)} "
        f"usable={int(usable.sum())} "
        f"missing_output="
        f"{int(df['output_profile_missing'].sum())} "
        f"missing_inject="
        f"{int(df['inject_profile_missing'].sum())}"
    )
PY

echo
echo "[DONE] Offline menu pipeline completed"