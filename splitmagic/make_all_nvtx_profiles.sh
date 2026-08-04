#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# 모든 *_nvtxsum.csv에서 다음 profile을 생성한다.
#
# 1. RECOMP/<op_type>/<node_name>
#    → *_recompute_layer_profile.csv
#    → *_recompute_layer_profile_details.csv
#
# 2. INJECT/Patch/<JIN key>
#    → *_inject_key_profile.csv
#    → *_inject_key_profile_details.csv
#
# Usage:
#   ./make_all_nvtx_profiles.sh [ROOT]
#
# Example:
#   ./make_all_nvtx_profiles.sh ./nsys_results
#
# Optional:
#   METRIC=median FORCE=1 ./make_all_nvtx_profiles.sh
#   METRIC=avg    FORCE=1 ./make_all_nvtx_profiles.sh
# ============================================================

ROOT="${1:-./nsys_results}"
METRIC="${METRIC:-median}"
FORCE="${FORCE:-0}"

PYTHON_BIN="${PYTHON_BIN:-/home/syu23/miniconda3/envs/torch-build/bin/python3}"

case "${METRIC}" in
    avg|median)
        ;;
    *)
        echo "[ERROR] METRIC must be avg or median" >&2
        exit 1
        ;;
esac

if [[ ! -d "${ROOT}" ]]; then
    echo "[ERROR] directory not found: ${ROOT}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

ROOT="$(realpath "${ROOT}")"

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] METRIC=${METRIC}"
echo "[INFO] FORCE=${FORCE}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"

count=0
success=0
failed=0
skipped=0
no_recompute=0
no_inject=0

while IFS= read -r -d '' input_csv; do
    count=$((count + 1))

    filename="$(basename "${input_csv}")"
    base="${filename%_nvtxsum.csv}"
    output_dir="$(dirname "${input_csv}")"

    # Bash에서 ${...}를 줄바꿈하면 안 되므로 올바른 값으로 재지정
    recompute_profile="${output_dir}/${base}_recompute_layer_profile.csv"
    recompute_details="${output_dir}/${base}_recompute_layer_profile_details.csv"
    inject_profile="${output_dir}/${base}_inject_key_profile.csv"
    inject_details="${output_dir}/${base}_inject_key_profile_details.csv"
    output_profile="${output_dir}/${base}_output_to_cpu_key_profile.csv"
    output_details="${output_dir}/${base}_output_to_cpu_key_profile_details.csv"

    echo
    echo "============================================================"
    echo "[FILE] ${input_csv}"
    echo "============================================================"

    if [[ "${FORCE}" != "1" ]] &&
        [[ -s "${recompute_profile}" ]] &&
        [[ -s "${inject_profile}" ]] &&
        [[ -s "${output_profile}" ]]; then

        echo "[SKIP] all profile outputs already exist"
        skipped=$((skipped + 1))
        continue
    fi

    if "${PYTHON_BIN}" - \
        "${input_csv}" \
        "${recompute_profile}" \
        "${recompute_details}" \
        "${inject_profile}" \
        "${inject_details}" \
        "${output_profile}" \
        "${output_details}" \
        "${METRIC}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any


input_csv = Path(sys.argv[1])
recompute_profile_path = Path(sys.argv[2])
recompute_details_path = Path(sys.argv[3])
inject_profile_path = Path(sys.argv[4])
inject_details_path = Path(sys.argv[5])
output_profile_path = Path(sys.argv[6])
output_details_path = Path(sys.argv[7])
metric = sys.argv[8]


def find_header_index(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        normalized = line.lstrip("\ufeff").strip()

        if normalized.startswith("Time (%)"):
            return index

    raise RuntimeError(
        "CSV header beginning with 'Time (%)' was not found"
    )


def parse_float(row: dict[str, str], name: str) -> float:
    value = row.get(name)

    if value is None or value == "":
        raise ValueError(f"missing value for column: {name}")

    return float(value)


def parse_instances(row: dict[str, str]) -> int:
    return int(float(row["Instances"]))


def make_common_metrics(
    row: dict[str, str],
) -> dict[str, Any]:
    avg_ms = parse_float(row, "Avg (ns)") / 1_000_000.0
    median_ms = parse_float(row, "Med (ns)") / 1_000_000.0
    min_ms = parse_float(row, "Min (ns)") / 1_000_000.0
    max_ms = parse_float(row, "Max (ns)") / 1_000_000.0
    total_ms = (
        parse_float(row, "Total Time (ns)")
        / 1_000_000.0
    )
    instances = parse_instances(row)

    selected_ms = (
        avg_ms
        if metric == "avg"
        else median_ms
    )

    return {
        "selected_ms": selected_ms,
        "avg_ms": avg_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "total_ms": total_ms,
        "instances": instances,
    }


lines = input_csv.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

header_index = find_header_index(lines)
reader = csv.DictReader(lines[header_index:])

recompute_rows: list[dict[str, Any]] = []
inject_rows: list[dict[str, Any]] = []
output_rows: list[dict[str, Any]] = []

for source_row in reader:
    range_name = (
        source_row.get("Range")
        or ""
    ).strip()

    if not range_name:
        continue

    try:
        metrics = make_common_metrics(source_row)
    except (TypeError, ValueError, KeyError) as exc:
        print(
            f"[WARN] invalid row skipped: "
            f"range={range_name} error={exc}",
            file=sys.stderr,
        )
        continue

    # ----------------------------------------------------
    # Recompute operator profile
    #
    # RECOMP/BN/layer1_0_bn1
    # ----------------------------------------------------
    if range_name.startswith("RECOMP/"):
        parts = range_name.split("/", 2)

        if len(parts) != 3:
            print(
                f"[WARN] invalid RECOMP range: {range_name}",
                file=sys.stderr,
            )
            continue

        _, op_type, node_name = parts

        recompute_rows.append({
            "node_name": node_name,
            "op_type": op_type,
            "range_name": range_name,
            **metrics,
        })

        continue

    # ----------------------------------------------------
    # Inject key profile
    #
    # INJECT/Patch/graph:relu:15:result
    # ----------------------------------------------------
    prefix = "INJECT/Patch/"

    if range_name.startswith(prefix):
        key = range_name[len(prefix):]

        if not key:
            print(
                f"[WARN] empty inject key: {range_name}",
                file=sys.stderr,
            )
            continue

        inject_rows.append({
            "key": key,
            "range_name": range_name,
            **metrics,
        })

        continue

    prefix = "OUTPUT_TO_CPU/"

    if range_name.startswith(prefix):

        key = range_name[len(prefix):]

        output_rows.append({
            "key": key,
            "range_name": range_name,
            **metrics,
        })

        continue


# ============================================================
# Recompute output
# ============================================================

if recompute_rows:
    recompute_rows.sort(
        key=lambda item: str(item["node_name"])
    )

    with recompute_profile_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "node_name",
                "avg_ms",
            ],
        )

        writer.writeheader()

        for item in recompute_rows:
            writer.writerow({
                "node_name": item["node_name"],
                # 기존 engine loader와 맞추기 위해
                # 선택된 metric을 avg_ms 열에 저장한다.
                "avg_ms": f"{item['selected_ms']:.9f}",
            })

    recompute_detail_fields = [
        "node_name",
        "op_type",
        "range_name",
        "selected_ms",
        "avg_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "total_ms",
        "instances",
    ]

    with recompute_details_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=recompute_detail_fields,
        )

        writer.writeheader()
        writer.writerows(recompute_rows)

    print(
        f"[RECOMPUTE] nodes={len(recompute_rows)} "
        f"profile={recompute_profile_path.name}"
    )

else:
    recompute_profile_path.unlink(missing_ok=True)
    recompute_details_path.unlink(missing_ok=True)

    print("[RECOMPUTE_EMPTY]")


# ============================================================
# Inject output
# ============================================================

if inject_rows:
    inject_rows.sort(
        key=lambda item: str(item["key"])
    )

    with inject_profile_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "key",
                "patch_ms",
            ],
        )

        writer.writeheader()

        for item in inject_rows:
            writer.writerow({
                "key": item["key"],
                "patch_ms": f"{item['selected_ms']:.9f}",
            })

    inject_detail_fields = [
        "key",
        "range_name",
        "selected_ms",
        "avg_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "total_ms",
        "instances",
    ]

    with inject_details_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=inject_detail_fields,
        )

        writer.writeheader()
        writer.writerows(inject_rows)

    print(
        f"[INJECT] keys={len(inject_rows)} "
        f"profile={inject_profile_path.name}"
    )

else:
    inject_profile_path.unlink(missing_ok=True)
    inject_details_path.unlink(missing_ok=True)

    print("[INJECT_EMPTY]")

# ============================================================
# OUTPUT_TO_CPU output
# ============================================================

if output_rows:

    output_rows.sort(
        key=lambda x: x["key"]
    )

    with output_profile_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "output_to_cpu_ms",
            ],
        )

        writer.writeheader()

        for row in output_rows:

            writer.writerow({
                "key": row["key"],
                "output_to_cpu_ms":
                    f"{row['selected_ms']:.9f}",
            })

    with output_details_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "range_name",
                "selected_ms",
                "avg_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "total_ms",
                "instances",
            ],
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"[OUTPUT_TO_CPU] "
        f"keys={len(output_rows)} "
        f"profile={output_profile_path.name}"
    )

else:

    output_profile_path.unlink(
        missing_ok=True
    )

    output_details_path.unlink(
        missing_ok=True
    )

    print("[OUTPUT_TO_CPU_EMPTY]")

if (
    not recompute_rows
    and not inject_rows
    and not output_rows
):
    raise SystemExit(3)
PY
    then
        success=$((success + 1))

        if [[ ! -s "${recompute_profile}" ]]; then
            no_recompute=$((no_recompute + 1))
        fi

        if [[ ! -s "${inject_profile}" ]]; then
            no_inject=$((no_inject + 1))
        fi
    else
        status=$?

        if [[ "${status}" -eq 3 ]]; then
            echo "[EMPTY] no RECOMP or INJECT/Patch ranges"
            no_recompute=$((no_recompute + 1))
            no_inject=$((no_inject + 1))
        else
            echo "[FAIL] ${input_csv}"
            failed=$((failed + 1))
        fi
    fi

done < <(
    PROFILE_RATIO="${PROFILE_RATIO:-}"

    if [[ -n "${PROFILE_RATIO}" ]]; then
        find "${ROOT}" \
            -type f \
            -path "*ratio${PROFILE_RATIO}*" \
            -name '*_nvtxsum.csv' \
            -print0
    else
        find "${ROOT}" \
            -type f \
            -name '*_nvtxsum.csv' \
            -print0
    fi
)

echo
echo "========================== SUMMARY =========================="
echo "input CSVs       : ${count}"
echo "successful       : ${success}"
echo "skipped          : ${skipped}"
echo "no RECOMP        : ${no_recompute}"
echo "no INJECT/Patch  : ${no_inject}"
echo "failed           : ${failed}"
echo "============================================================"

if [[ "${count}" -eq 0 ]]; then
    echo "[WARN] no *_nvtxsum.csv files found under ${ROOT}"
fi

if [[ "${failed}" -gt 0 ]]; then
    exit 2
fi