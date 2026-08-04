set -Eeuo pipefail

# ============================================================
# SplitMagic offline menu pipeline
#
# Conditions:
#   model       = resnet18
#   network     = 1000 Mbps
#   drop ratio  = 0.99
#   MPS         = 100 / 50 / 25 / 10
#   steps       = 1000
#   metric      = median
# ============================================================

PROJECT_ROOT="$PWD"
NSYS_ROOT="${PROJECT_ROOT}/nsys_results"
MERGED_ROOT="${PROJECT_ROOT}/merged_profiles"
MENU_ROOT="${PROJECT_ROOT}/menus_offline"

echo "============================================================"
echo "[1/6] Run profiling experiments"
echo "============================================================"

MAX_STEPS=1000 \
REPEATS=1 \
./run_all_local_nsys.sh

echo
echo "============================================================"
echo "[2/6] Convert qdstrm to nsys-rep"
echo "============================================================"

FORCE=1 \
./convert_all_qdstrm.sh

echo
echo "============================================================"
echo "[3/6] Export Nsight CSV reports"
echo "============================================================"

FORCE=1 \
./export_nsys_csv.sh

echo
echo "============================================================"
echo "[4/6] Extract ratio=0.99 NVTX profiles"
echo "============================================================"

PROFILE_RATIO=0.99 \
METRIC=median \
FORCE=1 \
./make_all_nvtx_profiles.sh \
    "${NSYS_ROOT}"

echo
echo "============================================================"
echo "[5/6] Merge ratio=0.99 profiles by MPS condition"
echo "============================================================"

PROFILE_RATIO=0.99 \
FORCE=1 \
./merge_all_nvtx_profiles.sh \
    "${NSYS_ROOT}" \
    "${MERGED_ROOT}"

echo
echo "============================================================"
echo "[6/6] Build final offline menus"
echo "============================================================"

FORCE=1 \
NETWORK_MBPS=1000 \
PROFILE_STEPS=1000 \
METRIC=median \
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

python3 - <<'PY'
from pathlib import Path

import pandas as pd


paths = sorted(
    Path("./menus_offline").glob(
        "resnet18_cuda_mps*/final_menu.csv"
    )
)

if not paths:
    raise RuntimeError(
        "No final_menu.csv files were generated"
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