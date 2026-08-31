# python3 plot_resnet18_results_by_dataset.py \
#     --gpu ./resnet18_gpu_summary.csv \
#     --cpu ./resnet18_cpu_summary.csv \
#     --output ./resnet18_result_figures

#!/usr/bin/env python3
"""
Create ResNet-18 result figures and compact result tables from GPU/CPU summaries.

Input:
  --gpu resnet18_gpu_summary.csv
  --cpu resnet18_cpu_summary.csv

Output layout:
  <output>/
    CIFAR10/
      GPU/
      CPU/
      tables/
    TinyImageNet/
      GPU/
      CPU/
      tables/

The script automatically:
- separates CIFAR-10 and Tiny-ImageNet,
- separates GPU ratio sweeps by selection policy and MPS setting,
- extracts CPU thread count from experiment names such as "threads4",
- uses error bars when matching *_std columns exist,
- produces compact CSV tables for the report.

Example:
  python3 plot_resnet18_results.py \
      --gpu ./resnet18_gpu_summary.csv \
      --cpu ./resnet18_cpu_summary.csv \
      --output ./resnet18_result_figures
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = {
    "cifar10": "CIFAR10",
    "tiny-imagenet": "TinyImageNet",
}

GPU_METRICS = {
    "payload": {
        "y": "log_payload_mb_mean",
        "err": "log_payload_mb_std",
        "ylabel": "Payload Size (MB)",
        "title": "Payload Size vs Drop Ratio",
        "filename": "payload_vs_ratio.png",
    },
    "communication": {
        "y": "log_application_communication_ms_mean",
        "err": "log_application_communication_ms_std",
        "ylabel": "Communication Time (ms)",
        "title": "Application Communication Time vs Drop Ratio",
        "filename": "communication_vs_ratio.png",
    },
    "recompute_inject": {
        "series": [
            ("b_recompute_total_ms", "Recomputation"),
            ("b_inject_all_ms", "Injection"),
        ],
        "ylabel": "Time (ms)",
        "title": "Recomputation and Injection vs Drop Ratio",
        "filename": "recompute_injection_vs_ratio.png",
    },
    "backward": {
        "series": [
            ("b_backward_jin_ms", "JIN backward"),
            ("b_torch_backward_ms", "torch backward"),
        ],
        "ylabel": "Time (ms)",
        "title": "Backward Processing Time vs Drop Ratio",
        "filename": "backward_vs_ratio.png",
    },
    "total": {
        "y": "log_total_ms_mean",
        "err": "log_total_ms_std",
        "ylabel": "End-to-End Step Time (ms)",
        "title": "End-to-End Step Time vs Drop Ratio",
        "filename": "total_time_vs_ratio.png",
    },
}

CPU_METRICS = {
    "payload": {
        "y": "log_payload_mb_mean",
        "err": "log_payload_mb_std",
        "ylabel": "Payload Size (MB)",
        "title": "Payload Size vs Drop Ratio",
        "filename": "payload_vs_ratio.png",
        "single_line": True,
    },
    "recompute": {
        "y": "b_recompute_total_ms",
        "ylabel": "Recomputation Time (ms)",
        "title": "Recomputation Time by CPU Thread Count",
        "filename": "recompute_by_threads.png",
    },
    "backward": {
        "y": "b_torch_backward_ms",
        "ylabel": "Backward Time (ms)",
        "title": "Backward Time by CPU Thread Count",
        "filename": "backward_by_threads.png",
    },
    "communication": {
        "y": "log_application_communication_ms_mean",
        "err": "log_application_communication_ms_std",
        "ylabel": "Communication Time (ms)",
        "title": "Communication Time by CPU Thread Count",
        "filename": "communication_by_threads.png",
    },
    "total": {
        "y": "log_total_ms_mean",
        "err": "log_total_ms_std",
        "ylabel": "End-to-End Step Time (ms)",
        "title": "End-to-End Step Time by CPU Thread Count",
        "filename": "total_time_by_threads.png",
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ResNet-18 SplitMagic GPU/CPU experiment summaries."
    )
    parser.add_argument("--gpu", type=Path, required=True, help="GPU summary CSV")
    parser.add_argument("--cpu", type=Path, required=True, help="CPU summary CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./resnet18_result_figures"),
        help="Output directory",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution",
    )
    return parser.parse_args()


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} summary not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} summary is empty: {path}")

    for col in [
        "ratio",
        "mps",
        "run_id",
        "network_mbps",
        "latency_ms",
        "log_payload_mb_mean",
        "log_payload_mb_std",
        "log_application_communication_ms_mean",
        "log_application_communication_ms_std",
        "log_total_ms_mean",
        "log_total_ms_std",
        "b_recompute_total_ms",
        "b_inject_all_ms",
        "b_backward_jin_ms",
        "b_torch_backward_ms",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_dataset(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "cifar": "cifar10",
        "cifar-10": "cifar10",
        "cifar10": "cifar10",
        "imagenet": "tiny-imagenet",
        "tiny_imagenet": "tiny-imagenet",
        "tinyimagenet": "tiny-imagenet",
        "tiny-imagenet": "tiny-imagenet",
    }
    return aliases.get(text, text)


def add_cpu_threads(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "threads" in result.columns:
        result["threads"] = pd.to_numeric(result["threads"], errors="coerce")
        return result

    if "experiment_name" not in result.columns:
        result["threads"] = np.nan
        return result

    result["threads"] = pd.to_numeric(
        result["experiment_name"]
        .astype(str)
        .str.extract(r"threads(\d+)", expand=False),
        errors="coerce",
    )
    return result


def save_figure(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def available(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(col in df.columns and df[col].notna().any() for col in columns)


def group_label(policy: object, mps: object) -> str:
    policy_text = str(policy) if pd.notna(policy) else "unknown"
    mps_text = f"MPS {int(mps)}%" if pd.notna(mps) else "MPS unknown"
    return f"{policy_text}, {mps_text}"


def aggregate_repeated_runs(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """
    If there are repeated run IDs for one configuration, average them.
    Existing per-step standard deviations remain in their own columns.
    """
    existing_groups = [c for c in group_cols if c in df.columns]
    existing_metrics = [c for c in metric_cols if c in df.columns]

    if not existing_groups or not existing_metrics:
        return df.copy()

    agg = {c: "mean" for c in existing_metrics}
    passthrough = [
        "experiment_name",
        "dataset",
        "selection_policy",
        "mps",
        "threads",
        "network_mbps",
        "latency_ms",
    ]
    for col in passthrough:
        if col in df.columns and col not in existing_groups:
            agg[col] = "first"

    return df.groupby(existing_groups, dropna=False, as_index=False).agg(agg)


# ---------------------------------------------------------------------------
# GPU plots
# ---------------------------------------------------------------------------

def gpu_ratio_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """
    Return only true ratio sweeps. A group must have at least three distinct
    ratio values. This prevents the old fixed-ratio MPS experiment from being
    mixed into ratio-sweep figures.
    """
    keys = [c for c in ["selection_policy", "mps", "network_mbps", "latency_ms"]
            if c in df.columns]

    groups: list[tuple[str, pd.DataFrame]] = []
    if not keys:
        return [("all", df)] if df["ratio"].nunique() >= 3 else []

    for key, group in df.groupby(keys, dropna=False):
        if group["ratio"].nunique() < 3:
            continue

        if not isinstance(key, tuple):
            key = (key,)
        values = dict(zip(keys, key))
        label = group_label(
            values.get("selection_policy"),
            values.get("mps"),
        )
        groups.append((label, group.sort_values("ratio")))

    return groups


def plot_gpu_single_metric(
    groups: list[tuple[str, pd.DataFrame]],
    spec: dict,
    output_path: Path,
    dpi: int,
) -> None:
    y = spec["y"]
    err = spec.get("err")

    valid_groups = [
        (label, group)
        for label, group in groups
        if available(group, ["ratio", y])
    ]
    if not valid_groups:
        return

    plt.figure(figsize=(8.5, 5.2))

    for label, group in valid_groups:
        x = group["ratio"]
        values = group[y]

        if err and err in group.columns and group[err].notna().any():
            plt.errorbar(
                x,
                values,
                yerr=group[err],
                marker="o",
                capsize=3,
                label=label,
            )
        else:
            plt.plot(x, values, marker="o", label=label)

    plt.xlabel("Configured Drop Ratio")
    plt.ylabel(spec["ylabel"])
    plt.title(spec["title"])
    plt.grid(True)
    if len(valid_groups) > 1:
        plt.legend()
    save_figure(output_path, dpi)


def plot_gpu_multi_metric(
    groups: list[tuple[str, pd.DataFrame]],
    spec: dict,
    output_path: Path,
    dpi: int,
) -> None:
    """
    Make one figure per ratio-sweep group when the figure contains several
    internal metrics. This avoids confusing policy/MPS labels with metric labels.
    """
    for label, group in groups:
        valid_series = [
            (column, series_label)
            for column, series_label in spec["series"]
            if available(group, ["ratio", column])
        ]
        if not valid_series:
            continue

        suffix = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
        path = output_path.with_name(
            f"{output_path.stem}_{suffix}{output_path.suffix}"
        )

        plt.figure(figsize=(8.5, 5.2))
        for column, series_label in valid_series:
            plt.plot(
                group["ratio"],
                group[column],
                marker="o",
                label=series_label,
            )

        plt.xlabel("Configured Drop Ratio")
        plt.ylabel(spec["ylabel"])
        plt.title(f"{spec['title']}\n({label})")
        plt.grid(True)
        plt.legend()
        save_figure(path, dpi)


def plot_gpu_mps_experiment(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """
    Plot configurations where one ratio has at least two distinct MPS values.
    """
    if not available(df, ["ratio", "mps"]):
        return

    for ratio, group in df.groupby("ratio"):
        if group["mps"].nunique() < 2:
            continue

        group = group.sort_values("mps")

        metrics = [
            ("b_recompute_total_ms", "Recomputation Time (ms)", "mps_recompute.png"),
            ("b_backward_jin_ms", "JIN Backward Time (ms)", "mps_backward.png"),
            ("log_total_ms_mean", "End-to-End Step Time (ms)", "mps_total_time.png"),
        ]

        ratio_dir = output_dir / f"MPS_Comparison_Ratio_{ratio:g}"
        for column, ylabel, filename in metrics:
            if not available(group, ["mps", column]):
                continue

            plt.figure(figsize=(7.5, 5))
            plt.plot(group["mps"], group[column], marker="o")
            plt.xlabel("CUDA MPS Active Thread Percentage")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel.replace(' (ms)', '')} vs GPU Capacity\nDrop Ratio = {ratio:g}")
            plt.grid(True)
            save_figure(ratio_dir / filename, dpi)


def make_gpu_tables(df: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "dataset",
        "selection_policy",
        "mps",
        "network_mbps",
        "latency_ms",
        "ratio",
        "actual_drop_ratio",
        "saved_tensor_count",
        "captured_tensor_mb",
        "dropped_count",
        "saved_mb",
        "log_payload_mb_mean",
        "b_recompute_total_ms",
        "b_inject_all_ms",
        "b_backward_jin_ms",
        "b_torch_backward_ms",
        "log_application_communication_ms_mean",
        "log_total_ms_mean",
        "log_total_ms_std",
        "log_loss_mean",
    ]
    selected = [c for c in columns if c in df.columns]
    table = df[selected].sort_values(
        [c for c in ["dataset", "selection_policy", "mps", "ratio"] if c in selected]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "gpu_result_table.csv", index=False)


# ---------------------------------------------------------------------------
# CPU plots
# ---------------------------------------------------------------------------

def plot_cpu_metric(
    df: pd.DataFrame,
    spec: dict,
    output_path: Path,
    dpi: int,
) -> None:
    y = spec["y"]
    err = spec.get("err")

    if not available(df, ["ratio", y]):
        return

    plt.figure(figsize=(8.5, 5.2))

    if spec.get("single_line"):
        # Payload is expected to be identical for all CPU thread counts.
        reduced = (
            df.groupby("ratio", as_index=False)[
                [c for c in [y, err] if c and c in df.columns]
            ]
            .mean()
            .sort_values("ratio")
        )
        if err and err in reduced.columns and reduced[err].notna().any():
            plt.errorbar(
                reduced["ratio"],
                reduced[y],
                yerr=reduced[err],
                marker="o",
                capsize=3,
            )
        else:
            plt.plot(reduced["ratio"], reduced[y], marker="o")
    else:
        if "threads" not in df.columns or df["threads"].isna().all():
            return

        for threads, group in df.groupby("threads"):
            group = group.sort_values("ratio")
            label = f"{int(threads)} threads"

            if err and err in group.columns and group[err].notna().any():
                plt.errorbar(
                    group["ratio"],
                    group[y],
                    yerr=group[err],
                    marker="o",
                    capsize=3,
                    label=label,
                )
            else:
                plt.plot(
                    group["ratio"],
                    group[y],
                    marker="o",
                    label=label,
                )
        plt.legend()

    plt.xlabel("Configured Drop Ratio")
    plt.ylabel(spec["ylabel"])
    plt.title(spec["title"])
    plt.grid(True)
    save_figure(output_path, dpi)


def plot_cpu_speedup(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    if not available(df, ["ratio", "threads", "log_total_ms_mean"]):
        return

    pivot = df.pivot_table(
        index="ratio",
        columns="threads",
        values="log_total_ms_mean",
        aggfunc="mean",
    ).sort_index()

    if 2 not in pivot.columns:
        return

    plt.figure(figsize=(8.5, 5.2))

    plotted = False
    for threads in sorted(c for c in pivot.columns if c != 2):
        speedup = pivot[2] / pivot[threads]
        if speedup.notna().any():
            plt.plot(
                speedup.index,
                speedup.values,
                marker="o",
                label=f"2 threads / {int(threads)} threads",
            )
            plotted = True

    if not plotted:
        plt.close()
        return

    plt.axhline(1.0, linewidth=1)
    plt.xlabel("Configured Drop Ratio")
    plt.ylabel("Speedup Relative to 2 Threads")
    plt.title("CPU Parallelism Speedup")
    plt.grid(True)
    plt.legend()
    save_figure(output_path, dpi)


def make_cpu_tables(df: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        "dataset",
        "threads",
        "network_mbps",
        "latency_ms",
        "ratio",
        "saved_tensor_count",
        "captured_tensor_mb",
        "log_payload_mb_mean",
        "b_recompute_total_ms",
        "b_inject_all_ms",
        "b_backward_jin_ms",
        "b_torch_backward_ms",
        "log_application_communication_ms_mean",
        "log_total_ms_mean",
        "log_total_ms_std",
        "log_loss_mean",
    ]
    selected = [c for c in columns if c in df.columns]
    table = df[selected].sort_values(
        [c for c in ["dataset", "threads", "ratio"] if c in selected]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "cpu_result_table.csv", index=False)

    if available(df, ["threads", "ratio", "log_total_ms_mean"]):
        best_rows = (
            df.dropna(subset=["threads", "ratio", "log_total_ms_mean"])
            .sort_values("log_total_ms_mean")
            .groupby(["dataset", "threads"], as_index=False)
            .first()
        )
        best_cols = [
            c for c in [
                "dataset",
                "threads",
                "ratio",
                "log_payload_mb_mean",
                "b_recompute_total_ms",
                "log_application_communication_ms_mean",
                "log_total_ms_mean",
                "log_total_ms_std",
            ]
            if c in best_rows.columns
        ]
        best_rows[best_cols].to_csv(
            output_dir / "cpu_best_configuration.csv",
            index=False,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    try:
        gpu = load_csv(args.gpu, "GPU")
        cpu = load_csv(args.cpu, "CPU")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if "dataset" not in gpu.columns or "dataset" not in cpu.columns:
        print("[ERROR] Both summaries must contain a dataset column.", file=sys.stderr)
        return 3

    gpu["dataset"] = gpu["dataset"].map(normalize_dataset)
    cpu["dataset"] = cpu["dataset"].map(normalize_dataset)
    cpu = add_cpu_threads(cpu)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # GPU figures: each dataset is processed independently.
    # No figure ever contains rows from both CIFAR-10 and Tiny-ImageNet.
    for dataset, dataset_df in gpu.groupby("dataset", dropna=False):
        if dataset not in DATASET_DIR:
            print(f"[WARN] Unrecognized GPU dataset skipped: {dataset}")
            continue

        dataset_dir = output / DATASET_DIR[dataset] / "GPU"
        ratio_groups = gpu_ratio_groups(dataset_df)

        for metric_name, spec in GPU_METRICS.items():
            target = dataset_dir / spec["filename"]

            if "series" in spec:
                plot_gpu_multi_metric(
                    ratio_groups,
                    spec,
                    target,
                    args.dpi,
                )
            else:
                plot_gpu_single_metric(
                    ratio_groups,
                    spec,
                    target,
                    args.dpi,
                )

        plot_gpu_mps_experiment(dataset_df, dataset_dir, args.dpi)

    # CPU figures: each dataset is processed independently.
    # Missing CPU data for a dataset simply produces no CPU directory/figures.
    for dataset, dataset_df in cpu.groupby("dataset", dropna=False):
        if dataset not in DATASET_DIR:
            print(f"[WARN] Unrecognized CPU dataset skipped: {dataset}")
            continue

        dataset_dir = output / DATASET_DIR[dataset] / "CPU"

        for spec in CPU_METRICS.values():
            plot_cpu_metric(
                dataset_df,
                spec,
                dataset_dir / spec["filename"],
                args.dpi,
            )

        plot_cpu_speedup(
            dataset_df,
            dataset_dir / "thread_speedup.png",
            args.dpi,
        )

    # Result tables are also kept strictly separated by dataset.
    for dataset in sorted(set(gpu["dataset"].dropna()) | set(cpu["dataset"].dropna())):
        if dataset not in DATASET_DIR:
            continue

        table_dir = output / DATASET_DIR[dataset] / "tables"

        gpu_dataset = gpu[gpu["dataset"] == dataset].copy()
        cpu_dataset = cpu[cpu["dataset"] == dataset].copy()

        if not gpu_dataset.empty:
            make_gpu_tables(gpu_dataset, table_dir)

        if not cpu_dataset.empty:
            make_cpu_tables(cpu_dataset, table_dir)

    print(f"[DONE] Results written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
