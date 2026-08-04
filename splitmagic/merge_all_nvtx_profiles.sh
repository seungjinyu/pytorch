#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Merge per-experiment recompute/inject/output-to-CPU
# profiles into one profile per model + device resource
# condition.
#
# Input:
#   *_recompute_layer_profile_details.csv
#   *_inject_key_profile_details.csv
#   *_output_to_cpu_key_profile_details.csv
# Output:
#   merged_profiles/
#     resnet18_cuda_mps100_recompute_profile.csv
#     resnet18_cuda_mps100_recompute_profile_details.csv
#     resnet18_cuda_mps100_inject_profile.csv
#     resnet18_cuda_mps100_inject_profile_details.csv
#     resnet18_cuda_mps100_output_to_cpu_profile.csv
#     resnet18_cuda_mps100_output_to_cpu_profile_details.csv
#
# Usage:
#   ./merge_all_nvtx_profiles.sh [ROOT] [OUTPUT_DIR]
#
# Example:
#   FORCE=1 ./merge_all_nvtx_profiles.sh \
#       ./nsys_results \
#       ./merged_profiles
# ============================================================

ROOT="${1:-./nsys_results}"
OUTPUT_DIR="${2:-./merged_profiles}"
FORCE="${FORCE:-0}"

# 비어 있으면 모든 ratio를 병합한다.
# 메뉴 생성용으로 PROFILE_RATIO=0.99를 지정하면
# ratio0.99 디렉터리의 결과만 병합한다.
PROFILE_RATIO="${PROFILE_RATIO:-}"

PYTHON_BIN="${PYTHON_BIN:-/home/syu23/miniconda3/envs/torch-build/bin/python3}"

if [[ ! -d "${ROOT}" ]]; then
    echo "[ERROR] directory not found: ${ROOT}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

ROOT="$(realpath "${ROOT}")"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] FORCE=${FORCE}"
echo "[INFO] PROFILE_RATIO=${PROFILE_RATIO:-all}"

"${PYTHON_BIN}" - \
    "${ROOT}" \
    "${OUTPUT_DIR}" \
    "${FORCE}" \
    "${PROFILE_RATIO}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
force = sys.argv[3] == "1"

profile_ratio = sys.argv[4].strip()

ratio_marker = (
    f"ratio{profile_ratio}"
    if profile_ratio
    else None
)

print(
    "[MERGE_FILTER] "
    f"profile_ratio={profile_ratio or 'all'} "
    f"marker={ratio_marker or 'none'}",
    flush=True,
)


MPS_RE = re.compile(r"mps(?P<value>\d+)")
THREAD_RE = re.compile(r"threads?(?P<value>\d+)")


def infer_model(path: Path) -> str:
    text = str(path).lower()

    if "resnet18" in text or "resnet" in text:
        return "resnet18"

    if "vgg11" in text or "vgg" in text:
        return "vgg11bn"

    if "mobilenet" in text:
        return "mobilenetv2"

    return "unknown"

def infer_resource(
    path: Path,
) -> tuple[str, str, int | None]:

    # path:
    # experiment_dir/nsys_csv/profile.csv
    experiment_name = (
        path.parent.parent.name.lower()
    )

    thread_match = THREAD_RE.search(
        experiment_name
    )

    if (
        "_cpu_" in experiment_name
        or thread_match is not None
    ):
        value = (
            int(thread_match.group("value"))
            if thread_match is not None
            else None
        )

        return "cpu", "threads", value

    mps_match = MPS_RE.search(
        experiment_name
    )

    value = (
        int(mps_match.group("value"))
        if mps_match is not None
        else None
    )

    return "cuda", "mps", value


def condition_key(path: Path) -> tuple[str, str, str, int]:
    model = infer_model(path)
    device, resource_name, resource_value = infer_resource(path)

    if model == "unknown":
        raise RuntimeError(
            f"Cannot infer model from path: {path}"
        )

    if resource_value is None:
        raise RuntimeError(
            f"Cannot infer resource value from path: {path}"
        )

    return (
        model,
        device,
        resource_name,
        resource_value,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def to_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


# ------------------------------------------------------------
# Group input files by model/device/resource
# ------------------------------------------------------------

recompute_files: dict[
    tuple[str, str, str, int],
    list[Path],
] = defaultdict(list)

inject_files: dict[
    tuple[str, str, str, int],
    list[Path],
] = defaultdict(list)

output_files: dict[
    tuple[str, str, str, int],
    list[Path],
] = defaultdict(list)

def ratio_matches(path: Path) -> bool:
    """
    PROFILE_RATIO가 비어 있으면 모든 실험을 허용한다.

    PROFILE_RATIO=0.99이면 경로에 ratio0.99가 포함된
    실험 결과만 허용한다.
    """
    if ratio_marker is None:
        return True

    return ratio_marker in str(path)


for path in sorted(
    root.rglob(
        "*_recompute_layer_profile_details.csv"
    )
):
    if not ratio_matches(path):
        continue

    recompute_files[
        condition_key(path)
    ].append(path)


for path in sorted(
    root.rglob(
        "*_inject_key_profile_details.csv"
    )
):
    if not ratio_matches(path):
        continue

    inject_files[
        condition_key(path)
    ].append(path)


for path in sorted(
    root.rglob(
        "*_output_to_cpu_key_profile_details.csv"
    )
):
    if not ratio_matches(path):
        continue

    output_files[
        condition_key(path)
    ].append(path)


conditions = sorted(
    set(recompute_files)
    | set(inject_files)
    | set(output_files)
)

print(
    f"[INFO] conditions={len(conditions)} "
    f"recompute_files="
    f"{sum(map(len, recompute_files.values()))} "
    f"inject_files="
    f"{sum(map(len, inject_files.values()))} "
    f"output_files="
    f"{sum(map(len, output_files.values()))}"
)


for condition in conditions:
    model, device, resource_name, resource_value = condition

    condition_name = (
        f"{model}_{device}_"
        f"{resource_name}{resource_value}"
    )

    print()
    print("=" * 60)
    print(f"[CONDITION] {condition_name}")
    print("=" * 60)

    # ========================================================
    # Recompute profile merge
    # ========================================================

    recomp_inputs = recompute_files.get(condition, [])

    if recomp_inputs:
        by_node: dict[
            tuple[str, str, str],
            dict[str, Any],
        ] = {}

        for path in recomp_inputs:
            for row in read_csv(path):
                node_name = row["node_name"]
                op_type = row["op_type"]
                range_name = row["range_name"]

                key = (
                    node_name,
                    op_type,
                    range_name,
                )

                item = by_node.setdefault(
                    key,
                    {
                        "node_name": node_name,
                        "op_type": op_type,
                        "range_name": range_name,
                        "total_ms": 0.0,
                        "instances": 0,
                        "min_ms": float("inf"),
                        "max_ms": float("-inf"),
                        "source_profiles": 0,
                        "medians": [],
                    },
                )

                item["total_ms"] += to_float(
                    row,
                    "total_ms",
                )

                item["instances"] += to_int(
                    row,
                    "instances",
                )

                item["min_ms"] = min(
                    item["min_ms"],
                    to_float(row, "min_ms"),
                )

                item["max_ms"] = max(
                    item["max_ms"],
                    to_float(row, "max_ms"),
                )

                item["source_profiles"] += 1
                item["medians"].append(
                    to_float(row, "median_ms")
                )

        rows = []

        for item in by_node.values():
            instances = item["instances"]

            weighted_avg_ms = (
                item["total_ms"] / instances
                if instances > 0
                else 0.0
            )

            medians = sorted(item["medians"])
            count = len(medians)

            if count == 0:
                median_of_medians = 0.0
            elif count % 2 == 1:
                median_of_medians = medians[count // 2]
            else:
                median_of_medians = (
                    medians[count // 2 - 1]
                    + medians[count // 2]
                ) / 2.0

            rows.append({
                "node_name": item["node_name"],
                "op_type": item["op_type"],
                "range_name": item["range_name"],
                "avg_ms": weighted_avg_ms,
                "median_ms": median_of_medians,
                "min_ms": item["min_ms"],
                "max_ms": item["max_ms"],
                "total_ms": item["total_ms"],
                "instances": instances,
                "source_profiles": item[
                    "source_profiles"
                ],
            })

        rows.sort(
            key=lambda row: row["node_name"]
        )

        simple_path = (
            output_dir
            / f"{condition_name}_recompute_profile.csv"
        )

        details_path = (
            output_dir
            / f"{condition_name}_recompute_profile_details.csv"
        )

        if force or not simple_path.exists():
            with simple_path.open(
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

                for row in rows:
                    writer.writerow({
                        "node_name": row["node_name"],
                        "avg_ms": (
                            f"{row['median_ms']:.9f}"
                        ),
                    })

        if force or not details_path.exists():
            with details_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "node_name",
                        "op_type",
                        "range_name",
                        "avg_ms",
                        "median_ms",
                        "min_ms",
                        "max_ms",
                        "total_ms",
                        "instances",
                        "source_profiles",
                    ],
                )

                writer.writeheader()
                writer.writerows(rows)

        print(
            f"[RECOMPUTE] inputs={len(recomp_inputs)} "
            f"nodes={len(rows)} "
            f"output={simple_path.name}"
        )

    else:
        print("[RECOMPUTE] no inputs")

    # ========================================================
    # Inject profile merge
    # ========================================================

    inject_inputs = inject_files.get(condition, [])

    if inject_inputs:
        by_key: dict[str, dict[str, Any]] = {}

        for path in inject_inputs:
            for row in read_csv(path):
                jin_key = row["key"]

                item = by_key.setdefault(
                    jin_key,
                    {
                        "key": jin_key,
                        "range_name": row[
                            "range_name"
                        ],
                        "total_ms": 0.0,
                        "instances": 0,
                        "min_ms": float("inf"),
                        "max_ms": float("-inf"),
                        "source_profiles": 0,
                        "medians": [],
                    },
                )

                item["total_ms"] += to_float(
                    row,
                    "total_ms",
                )

                item["instances"] += to_int(
                    row,
                    "instances",
                )

                item["min_ms"] = min(
                    item["min_ms"],
                    to_float(row, "min_ms"),
                )

                item["max_ms"] = max(
                    item["max_ms"],
                    to_float(row, "max_ms"),
                )

                item["source_profiles"] += 1
                item["medians"].append(
                    to_float(row, "median_ms")
                )

        rows = []

        for item in by_key.values():
            instances = item["instances"]

            weighted_avg_ms = (
                item["total_ms"] / instances
                if instances > 0
                else 0.0
            )

            medians = sorted(item["medians"])
            count = len(medians)

            if count == 0:
                median_of_medians = 0.0
            elif count % 2 == 1:
                median_of_medians = medians[count // 2]
            else:
                median_of_medians = (
                    medians[count // 2 - 1]
                    + medians[count // 2]
                ) / 2.0

            rows.append({
                "key": item["key"],
                "range_name": item["range_name"],
                "patch_ms": weighted_avg_ms,
                "median_ms": median_of_medians,
                "min_ms": item["min_ms"],
                "max_ms": item["max_ms"],
                "total_ms": item["total_ms"],
                "instances": instances,
                "source_profiles": item[
                    "source_profiles"
                ],
            })

        rows.sort(
            key=lambda row: row["key"]
        )

        simple_path = (
            output_dir
            / f"{condition_name}_inject_profile.csv"
        )

        details_path = (
            output_dir
            / f"{condition_name}_inject_profile_details.csv"
        )

        if force or not simple_path.exists():
            with simple_path.open(
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

                for row in rows:
                    writer.writerow({
                        "key": row["key"],
                        "patch_ms": (
                            f"{row['median_ms']:.9f}"
                        ),
                    })

        if force or not details_path.exists():
            with details_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "key",
                        "range_name",
                        "patch_ms",
                        "median_ms",
                        "min_ms",
                        "max_ms",
                        "total_ms",
                        "instances",
                        "source_profiles",
                    ],
                )

                writer.writeheader()
                writer.writerows(rows)

        print(
            f"[INJECT] inputs={len(inject_inputs)} "
            f"keys={len(rows)} "
            f"output={simple_path.name}"
        )

    else:
        print("[INJECT] no inputs")

    # ========================================================
    # Output-to-CPU profile merge
    # ========================================================

    output_inputs = output_files.get(
        condition,
        [],
    )

    if output_inputs:
        by_key: dict[str, dict[str, Any]] = {}

        for path in output_inputs:
            for row in read_csv(path):
                jin_key = row["key"]

                item = by_key.setdefault(
                    jin_key,
                    {
                        "key": jin_key,
                        "range_name": row[
                            "range_name"
                        ],
                        "total_ms": 0.0,
                        "instances": 0,
                        "min_ms": float("inf"),
                        "max_ms": float("-inf"),
                        "source_profiles": 0,
                        "medians": [],
                    },
                )

                item["total_ms"] += to_float(
                    row,
                    "total_ms",
                )

                item["instances"] += to_int(
                    row,
                    "instances",
                )

                item["min_ms"] = min(
                    item["min_ms"],
                    to_float(row, "min_ms"),
                )

                item["max_ms"] = max(
                    item["max_ms"],
                    to_float(row, "max_ms"),
                )

                item["source_profiles"] += 1

                item["medians"].append(
                    to_float(
                        row,
                        "median_ms",
                    )
                )

        rows = []

        for item in by_key.values():
            instances = item["instances"]

            weighted_avg_ms = (
                item["total_ms"] / instances
                if instances > 0
                else 0.0
            )

            medians = sorted(
                item["medians"]
            )
            count = len(medians)

            if count == 0:
                median_of_medians = 0.0
            elif count % 2 == 1:
                median_of_medians = (
                    medians[count // 2]
                )
            else:
                median_of_medians = (
                    medians[count // 2 - 1]
                    + medians[count // 2]
                ) / 2.0

            rows.append({
                "key": item["key"],
                "range_name": item[
                    "range_name"
                ],
                "output_to_cpu_ms": (
                    weighted_avg_ms
                ),
                "median_ms": (
                    median_of_medians
                ),
                "min_ms": item["min_ms"],
                "max_ms": item["max_ms"],
                "total_ms": item["total_ms"],
                "instances": instances,
                "source_profiles": item[
                    "source_profiles"
                ],
            })

        rows.sort(
            key=lambda row: row["key"]
        )

        simple_path = (
            output_dir
            / (
                f"{condition_name}"
                f"_output_to_cpu_profile.csv"
            )
        )

        details_path = (
            output_dir
            / (
                f"{condition_name}"
                f"_output_to_cpu_profile_details.csv"
            )
        )

        if force or not simple_path.exists():
            with simple_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "key",
                        "output_to_cpu_ms",
                    ],
                )

                writer.writeheader()

                for row in rows:
                    writer.writerow({
                        "key": row["key"],
                        "output_to_cpu_ms": (
                            f"{row['median_ms']:.9f}"
                        ),
                    })

        if force or not details_path.exists():
            with details_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "key",
                        "range_name",
                        "output_to_cpu_ms",
                        "median_ms",
                        "min_ms",
                        "max_ms",
                        "total_ms",
                        "instances",
                        "source_profiles",
                    ],
                )

                writer.writeheader()
                writer.writerows(rows)

        print(
            f"[OUTPUT_TO_CPU] "
            f"inputs={len(output_inputs)} "
            f"keys={len(rows)} "
            f"output={simple_path.name}"
        )

    else:
        print("[OUTPUT_TO_CPU] no inputs")
PY

echo
echo "[DONE] merged profiles saved to: ${OUTPUT_DIR}"