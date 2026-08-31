# python3 make_gpu_summary.py \
#     ./nsys_results \
#     --device gpu \
#     -o resnet18_gpu_summary.csv


# python3 make_gpu_summary.py \
#     ./nsys_results \
#     --device cpu \
#     -o resnet18_cpu_summary.csv


#!/usr/bin/env python3
"""
Build one summary.csv from SplitMagic GPU Nsight experiment folders.

Expected per experiment:
  experiment.env
  node_a.log
  node_b.log
  nsys_csv/
    node_a_*_nvtxsum.csv
    node_b_*_nvtxsum.csv
    node_b_*_cudaapisum.csv       (optional)
    node_b_*_gpukernsum.csv        (optional)
    node_b_*_gpumemtimesum.csv     (optional)

Usage:
  python3 make_gpu_summary.py ./nsys_results -o ./gpu_summary.csv

Notes:
- Handles Nsight 2022.4 CSV files that begin with informational lines
  before the real CSV header.
- NVTX durations are converted from ns to ms.
- A_step_* rows are averaged after excluding the first N warm-up steps.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


NODE_A_STEP_RE = re.compile(
    rf"""
    ^\[Node\ A\]\s+
    epoch=(?P<epoch>\d+)\s+
    step=(?P<step>\d+)\s+
    loss=(?P<loss>{FLOAT})\s+
    payload_mb=(?P<payload_mb>{FLOAT})\s+
    capture_forward_ms=(?P<capture_forward_ms>{FLOAT})\s+
    alias_ms=(?P<alias_ms>{FLOAT})\s+
    auto_drop_ms=(?P<auto_drop_ms>{FLOAT})\s+
    send_recv_ms=(?P<send_recv_ms>{FLOAT})
    (?:\s+application_communication_ms=(?P<application_communication_ms>{FLOAT}))?
    (?:\s+state_load_ms=(?P<state_load_ms>{FLOAT}))?
    (?:\s+total_ms=(?P<total_ms>{FLOAT}))?
    """,
    re.VERBOSE,
)

CAPTURE_RE = re.compile(
    rf"\[Payload\]\[CAPTURE_ADD_TENSOR_PROFILE\]\s+"
    rf"count=(?P<count>\d+)\s+"
    rf"total_mb=(?P<total_mb>{FLOAT})"
)

ALIAS_RE = re.compile(
    rf"\[ALIAS\]\[SUMMARY\]\s+"
    rf"removed=(?P<removed>\d+)\s+"
    rf"saved_mb=(?P<saved_mb>{FLOAT})"
)

AUTO_DROP_RE = re.compile(
    rf"\[AUTO_DROP_(?:COST|RATIO)?\].*?"
    rf"dropped=(?P<dropped>\d+).*?"
    rf"drop_ratio=(?P<drop_ratio>{FLOAT}).*?"
    rf"saved_mb=(?P<saved_mb>{FLOAT})"
)

NODE_B_RECOMPUTE_RE = re.compile(
    rf"""
    \[B\]\[RECOMPUTE_INJECT_PROFILE\].*?
    n=(?P<recomputed_count>\d+).*?
    payload_mb=(?P<recomputed_mb>{FLOAT}).*?
    inject_mem_ms=(?P<inject_ms>{FLOAT})
    """,
    re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="Root containing experiment directories")
    p.add_argument("-o", "--output", type=Path, default=Path("gpu_summary.csv"))
    p.add_argument("--warmup", type=int, default=1,
                   help="Number of initial steps to exclude from averages")
    p.add_argument("--model-filter", default="resnet18")
    p.add_argument(
        "--device",
        choices=["gpu", "cpu", "all"],
        default="gpu",
        help="Select which experiment folders to parse."
    )
    return p.parse_args()

def device_match(exp_dir: Path, device: str) -> bool:
    name = exp_dir.name.lower()

    is_cpu = "_cpu_" in name

    if device == "gpu":
        return not is_cpu

    if device == "cpu":
        return is_cpu

    return True

def as_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def as_int(x: Any) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_env(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def folder_metadata(exp_dir: Path) -> dict[str, Any]:
    name = exp_dir.name
    low = name.lower()

    def find(pattern: str, cast):
        m = re.search(pattern, low)
        return cast(m.group(1)) if m else None
    if "imagenet" in low:
        dataset = "tiny-imagenet"
    else:
        dataset = "cifar10"

    return {
        "experiment_name": name,
        "experiment_dir": str(exp_dir),
        "model": "resnet18" if "resnet18" in low else None,
        "dataset": dataset,
        "network_mbps": find(r"_(\d+)mbps_", int),
        "latency_ms": find(r"_(\d+)ms_", int),
        "ratio": find(r"(?:maxratio|ratio)(\d+(?:\.\d+)?)", float),
        "mps": find(r"_mps(\d+)(?:_|$)", int),
        "run_id": find(r"_run(\d+)(?:_|$)", int),
        "selection_policy": (
            "cost" if "policycost" in low
            else "ratio" if "policyratio" in low or "_ratio" in low
            else None
        ),
    }


def merge_env(meta: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "run_id": ("RUN_ID", int),
        "network_mbps": ("NETWORK_MBPS", float),
        "latency_ms": ("LATENCY_MS", float),
        "ratio": ("DROP_RATIO", float),
        "mps": ("MPS_PERCENT", int),
        "selection_policy": ("SELECTION_POLICY", str),
        "max_steps": ("MAX_STEPS", int),
        "batch_size": ("BATCH_SIZE", int),
        "dataset": ("DATASET", str),
        "model": ("MODEL", str),
    }

    result = dict(meta)
    for dst, (src, cast) in aliases.items():

        if src not in env:
            continue

        value = str(env[src]).strip()

        # 빈 값이면 folder_metadata()에서 만든 값을 유지
        if value == "":
            continue

        try:
            result[dst] = cast(value)
        except ValueError:
            pass
    return result


def read_nsys_csv(path: Path) -> pd.DataFrame:
    """
    Nsight 2022.4 writes informational text before the actual CSV header.
    Find the first comma-separated header containing a recognized key.
    """
    lines = path.read_text(errors="replace").splitlines()
    header_index = None
    recognized = {"Time (%)", "Total Time (ns)", "Range", "Name", "Operation"}

    for i, line in enumerate(lines):
        try:
            cols = next(csv.reader([line]))
        except csv.Error:
            continue
        if len(cols) >= 2 and any(c.strip() in recognized for c in cols):
            if "Total Time (ns)" in [c.strip() for c in cols]:
                header_index = i
                break

    if header_index is None:
        return pd.DataFrame()

    try:
        return pd.read_csv(path, skiprows=header_index)
    except Exception:
        return pd.DataFrame()


def find_single(exp_dir: Path, pattern: str) -> Path | None:
    found = sorted((exp_dir / "nsys_csv").glob(pattern))
    return found[0] if found else None


def nvtx_row(df: pd.DataFrame, name: str) -> pd.Series | None:
    if df.empty or "Range" not in df.columns:
        return None
    rows = df[df["Range"] == name]
    return rows.iloc[0] if not rows.empty else None


def nvtx_avg_ms(df: pd.DataFrame, name: str) -> float | None:
    row = nvtx_row(df, name)
    if row is None:
        return None
    return as_float(row.get("Avg (ns)")) / 1e6


def nvtx_total_ms(df: pd.DataFrame, name: str) -> float | None:
    row = nvtx_row(df, name)
    if row is None:
        return None
    return as_float(row.get("Total Time (ns)")) / 1e6


def nvtx_instances(df: pd.DataFrame, name: str) -> int | None:
    row = nvtx_row(df, name)
    return None if row is None else as_int(row.get("Instances"))


def nvtx_prefix_total_ms(df: pd.DataFrame, prefix: str) -> float | None:
    if df.empty or "Range" not in df.columns or "Total Time (ns)" not in df.columns:
        return None
    rows = df[df["Range"].astype(str).str.startswith(prefix)]
    if rows.empty:
        return None
    return pd.to_numeric(rows["Total Time (ns)"], errors="coerce").sum() / 1e6


def nvtx_step_average_ms(df: pd.DataFrame, warmup: int) -> tuple[float | None, int]:
    if df.empty or "Range" not in df.columns:
        return None, 0

    rows = df[df["Range"].astype(str).str.match(r"^A_step_\d+$")].copy()
    if rows.empty:
        return None, 0

    rows["step_number"] = rows["Range"].str.extract(r"(\d+)$").astype(int)
    rows = rows[rows["step_number"] >= warmup]

    values = pd.to_numeric(rows["Avg (ns)"], errors="coerce").dropna()
    if values.empty:
        return None, 0

    return float(values.mean() / 1e6), int(values.count())


def parse_node_a_log(path: Path, warmup: int) -> dict[str, Any]:
    if not path.exists():
        return {}

    step_rows: list[dict[str, Any]] = []
    capture_records: list[dict[str, Any]] = []
    alias_records: list[dict[str, Any]] = []
    drop_records: list[dict[str, Any]] = []

    for line in path.read_text(errors="replace").splitlines():
        m = NODE_A_STEP_RE.match(line)
        if m:
            row = {k: as_float(v) for k, v in m.groupdict().items()}
            row["step"] = as_int(m.group("step"))
            row["epoch"] = as_int(m.group("epoch"))
            step_rows.append(row)
            continue

        m = CAPTURE_RE.search(line)
        if m:
            capture_records.append({
                "saved_tensor_count": as_int(m.group("count")),
                "captured_tensor_mb": as_float(m.group("total_mb")),
            })
            continue

        m = ALIAS_RE.search(line)
        if m:
            alias_records.append({
                "alias_removed_count": as_int(m.group("removed")),
                "alias_saved_mb": as_float(m.group("saved_mb")),
            })
            continue

        m = AUTO_DROP_RE.search(line)
        if m:
            drop_records.append({
                "dropped_count": as_int(m.group("dropped")),
                "actual_drop_ratio": as_float(m.group("drop_ratio")),
                "saved_mb": as_float(m.group("saved_mb")),
            })

    out: dict[str, Any] = {}

    if step_rows:
        df = pd.DataFrame(step_rows)
        df = df[df["step"] >= warmup]
        for col in [
            "loss", "payload_mb", "capture_forward_ms", "alias_ms",
            "auto_drop_ms", "send_recv_ms",
            "application_communication_ms", "state_load_ms", "total_ms"
        ]:
            if col in df.columns:
                out[f"log_{col}_mean"] = pd.to_numeric(df[col], errors="coerce").mean()
                out[f"log_{col}_std"] = pd.to_numeric(df[col], errors="coerce").std()
        out["log_step_count"] = len(df)

    for records, prefix in [
        (capture_records, ""),
        (alias_records, ""),
        (drop_records, ""),
    ]:
        if records:
            df = pd.DataFrame(records)
            if len(df) > warmup:
                df = df.iloc[warmup:]
            for col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").mean()

    return out


def parse_node_b_log(path: Path, warmup: int) -> dict[str, Any]:
    if not path.exists():
        return {}

    records: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = NODE_B_RECOMPUTE_RE.search(line)
        if m:
            records.append({
                "recomputed_count": as_int(m.group("recomputed_count")),
                "recomputed_mb": as_float(m.group("recomputed_mb")),
                "log_inject_ms": as_float(m.group("inject_ms")),
            })

    if not records:
        return {}

    df = pd.DataFrame(records)
    if len(df) > warmup:
        df = df.iloc[warmup:]

    return {
        col: pd.to_numeric(df[col], errors="coerce").mean()
        for col in df.columns
    }


def parse_cuda_api(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "Name" not in df.columns:
        return {}

    out: dict[str, Any] = {}
    total_ns = pd.to_numeric(df["Total Time (ns)"], errors="coerce").sum()
    out["cuda_api_total_ms"] = total_ns / 1e6

    for api in ["cudaMemcpyAsync", "cudaLaunchKernel", "cudaStreamSynchronize"]:
        rows = df[df["Name"] == api]
        if not rows.empty:
            out[f"{api}_total_ms"] = as_float(rows.iloc[0]["Total Time (ns)"]) / 1e6
            out[f"{api}_calls"] = as_int(rows.iloc[0].get("Num Calls"))
    return out


def parse_gpu_mem(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "Operation" not in df.columns:
        return {}

    mapping = {
        "[CUDA memcpy HtoD]": "gpu_mem_htod_ms",
        "[CUDA memcpy DtoH]": "gpu_mem_dtoh_ms",
        "[CUDA memcpy DtoD]": "gpu_mem_dtod_ms",
        "[CUDA memset]": "gpu_mem_memset_ms",
    }

    out: dict[str, Any] = {}
    for operation, col in mapping.items():
        rows = df[df["Operation"] == operation]
        if not rows.empty:
            out[col] = as_float(rows.iloc[0]["Total Time (ns)"]) / 1e6
            out[col.replace("_ms", "_count")] = as_int(rows.iloc[0].get("Count"))
    return out


def summarize_experiment(exp_dir: Path, warmup: int) -> dict[str, Any] | None:
    nsys_dir = exp_dir / "nsys_csv"
    if not nsys_dir.is_dir():
        return None

    a_nvtx_path = find_single(exp_dir, "node_a_*_nvtxsum.csv")
    b_nvtx_path = find_single(exp_dir, "node_b_*_nvtxsum.csv")

    if a_nvtx_path is None and b_nvtx_path is None:
        return None

    meta = merge_env(folder_metadata(exp_dir), parse_env(exp_dir / "experiment.env"))
    row: dict[str, Any] = dict(meta)

    a_nvtx = read_nsys_csv(a_nvtx_path) if a_nvtx_path else pd.DataFrame()
    b_nvtx = read_nsys_csv(b_nvtx_path) if b_nvtx_path else pd.DataFrame()

    # Node A NVTX metrics
    a_names = {
        "a_request_reply_ms": "A_request_reply_total",
        "a_wait_reply_ms": "A_wait_zmq_reply",
        "a_capture_forward_ms": "A_capture_forward",
        "a_alias_ms": "A_alias_duplicate",
        "a_selection_ratio_ms": "A_selection_ratio",
        "a_selection_cost_ms": "A_selection_cost",
        "a_serialize_ms": "A_serialize_payload_jin1",
        "a_build_message_ms": "A_build_zmq_message",
        "a_send_pyobj_ms": "A_zmq_send_pyobj",
        "a_state_load_ms": "A_load_updated_state",
        "a_payload_profile_ms": "A_payload_profile_print",
    }
    for col, nvtx_name in a_names.items():
        row[col] = nvtx_avg_ms(a_nvtx, nvtx_name)

    row["a_step_ms"], row["a_step_instances_used"] = nvtx_step_average_ms(a_nvtx, warmup)

    # Node B NVTX metrics
    b_names = {
        "b_wait_request_ms": "B_wait_request",
        "b_handle_request_ms": "B_handle_training_request",
        "b_backward_jin_ms": "B_backward_jin_total",
        "b_torch_backward_ms": "B_torch_backward",
        "b_recompute_total_ms": "B_recompute_total",
        "b_recompute_missing_ms": "B_recompute_missing_tensors",
        "b_recompute_execute_ms": "B_recompute_execute",
        "b_recompute_plan_ms": "B_recompute_plan",
        "b_inject_all_ms": "B_injected_recompute_tensors",
        "b_patch_all_ms": "INJECT/jin_patch_all",
        "b_dummy_forward_ms": "B_dummy_forward",
        "b_loss_build_ms": "B_loss_build",
        "b_deserialize_ms": "B_deserialize_jin1_payload",
        "b_reserialize_ms": "B_reserialize_payload_jin1",
        "b_optimizer_ms": "B_optimizer_step",
        "b_zero_grad_ms": "B_zero_grad",
        "b_state_dump_cpu_ms": "B_state_dump_to_cpu",
        "b_build_reply_ms": "B_build_reply",
        "b_send_reply_ms": "B_zmq_send_reply",
        "b_send_pyobj_reply_ms": "B_send_pyobj_reply",
        "b_load_state_ms": "B_load_state_dict",
    }
    for col, nvtx_name in b_names.items():
        row[col] = nvtx_avg_ms(b_nvtx, nvtx_name)

    # Summed detailed ranges over whole report, useful for diagnosis.
    row["recomp_conv_total_report_ms"] = nvtx_prefix_total_ms(b_nvtx, "RECOMP/Conv/")
    row["recomp_bn_total_report_ms"] = nvtx_prefix_total_ms(b_nvtx, "RECOMP/BN/")
    row["recomp_relu_total_report_ms"] = nvtx_prefix_total_ms(b_nvtx, "RECOMP/ReLU/")
    row["recomp_add_total_report_ms"] = nvtx_prefix_total_ms(b_nvtx, "RECOMP/Add/")
    row["inject_patch_total_report_ms"] = nvtx_prefix_total_ms(b_nvtx, "INJECT/Patch/")
    row["output_to_cpu_total_report_ms"] = nvtx_prefix_total_ms(b_nvtx, "OUTPUT_TO_CPU/")

    # Logs provide payload, loss and selection results.
    row.update(parse_node_a_log(exp_dir / "node_a.log", warmup))
    row.update(parse_node_b_log(exp_dir / "node_b.log", warmup))

    # Optional CUDA summaries.
    b_cuda_path = find_single(exp_dir, "node_b_*_cudaapisum.csv")
    b_mem_path = find_single(exp_dir, "node_b_*_gpumemtimesum.csv")

    if b_cuda_path:
        row.update(parse_cuda_api(read_nsys_csv(b_cuda_path)))
    if b_mem_path:
        row.update(parse_gpu_mem(read_nsys_csv(b_mem_path)))

    return row


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        return 2

    experiment_dirs = sorted({
        p.parent.parent
        for p in root.rglob("node_a_*_nvtxsum.csv")
        if p.parent.name == "nsys_csv"
        and args.model_filter.lower() in str(p.parent.parent).lower()
        and device_match(p.parent.parent, args.device)

    } | {
        p.parent.parent
        for p in root.rglob("node_b_*_nvtxsum.csv")
        if p.parent.name == "nsys_csv"
        and args.model_filter.lower() in str(p.parent.parent).lower()
        and device_match(p.parent.parent, args.device)
    })

    rows: list[dict[str, Any]] = []
    for exp_dir in experiment_dirs:
        row = summarize_experiment(exp_dir, args.warmup)
        if row:
            rows.append(row)
            print(f"[OK] {exp_dir.name}")

    if not rows:
        print("[ERROR] No matching experiment CSVs found.", file=sys.stderr)
        return 3

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["dataset", "network_mbps", "mps", "ratio", "run_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")
    print(f"Rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
