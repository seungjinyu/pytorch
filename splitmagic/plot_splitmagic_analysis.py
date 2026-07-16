#!/usr/bin/env python3
"""
Generate four SplitMagic analysis figures from:

1. recompute_layer_profile.csv
2. recompute_menu.csv
3. recompute_menu_resnet18_bs32_c1.csv
4. drop_ratio_experiment_all_networks.csv

Outputs:
  figures/01_predicted_vs_measured.png
  figures/02_candidate_efficiency.png
  figures/03_total_runtime_vs_drop_ratio.png
  figures/04_runtime_breakdown_<network>.png
  figures/cost_model_validation.csv
  figures/end_to_end_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def network_label(value: float) -> str:
    return "Unlimited" if float(value) == 0.0 else f"{value:g} Mbps"


def load_and_validate(
    layer_path: Path,
    estimated_path: Path,
    measured_path: Path,
    e2e_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer = pd.read_csv(layer_path)
    estimated = pd.read_csv(estimated_path)
    measured = pd.read_csv(measured_path)
    e2e = pd.read_csv(e2e_path)

    require_columns(
        layer,
        {"node_name", "fx_op", "op_type", "avg_ms"},
        layer_path.name,
    )
    require_columns(
        estimated,
        {
            "key",
            "tensor_mb",
            "start",
            "target",
            "path_len",
            "individual_cost_ms",
            "path",
        },
        estimated_path.name,
    )
    require_columns(
        measured,
        {
            "key",
            "tensor_mb",
            "start",
            "target",
            "path_len",
            "recompute_ms",
            "path",
        },
        measured_path.name,
    )
    require_columns(
        e2e,
        {
            "network_limit_mbps",
            "drop_ratio",
            "payload_mb",
            "estimated_grouped_ms",
            "recompute_wall_ms",
            "inject_ms",
            "torch_backward_ms",
            "request_round_trip_ms",
            "node_b_processing_ms",
            "node_a_total_ms",
        },
        e2e_path.name,
    )

    return layer, estimated, measured, e2e


def build_validation_table(
    estimated: pd.DataFrame,
    measured: pd.DataFrame,
) -> pd.DataFrame:
    # Only candidates present in both files can be validated.
    merged = estimated[
        [
            "key",
            "tensor_mb",
            "start",
            "target",
            "path_len",
            "individual_cost_ms",
            "path",
        ]
    ].merge(
        measured[
            [
                "key",
                "recompute_ms",
                "out_mb",
                "path",
            ]
        ].rename(columns={"path": "measured_path"}),
        on="key",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.rename(
        columns={
            "individual_cost_ms": "predicted_ms",
            "recompute_ms": "measured_ms",
        }
    )

    merged["absolute_error_ms"] = (
        merged["measured_ms"] - merged["predicted_ms"]
    ).abs()

    nonzero = merged["measured_ms"].replace(0, np.nan)
    merged["absolute_percentage_error"] = (
        merged["absolute_error_ms"] / nonzero * 100.0
    )

    merged["prediction_ratio"] = (
        merged["predicted_ms"]
        / merged["measured_ms"].replace(0, np.nan)
    )

    return merged.sort_values("measured_ms").reset_index(drop=True)


def print_validation_metrics(validation: pd.DataFrame) -> None:
    x = validation["predicted_ms"].to_numpy(dtype=float)
    y = validation["measured_ms"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(y - x)))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mape = float(
        np.nanmean(validation["absolute_percentage_error"].to_numpy(dtype=float))
    )
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(validation) > 1 else np.nan

    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_identity = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print("\n[COST MODEL VALIDATION]")
    print(f"Matched candidates : {len(validation)}")
    print(f"MAE                : {mae:.6f} ms")
    print(f"RMSE               : {rmse:.6f} ms")
    print(f"MAPE               : {mape:.2f}%")
    print(f"Pearson correlation: {correlation:.4f}")
    print(f"R² against y=x     : {r2_identity:.4f}")


def plot_predicted_vs_measured(
    validation: pd.DataFrame,
    out_path: Path,
) -> None:
    x = validation["predicted_ms"]
    y = validation["measured_ms"]

    limit = float(max(x.max(), y.max()) * 1.08)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, s=55, alpha=0.8)
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1.5)

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Predicted recompute cost (ms)")
    ax.set_ylabel("Measured recompute cost (ms)")
    ax.set_title("Predicted vs. measured recompute cost")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_candidate_efficiency(
    measured: pd.DataFrame,
    out_path: Path,
    annotate_top: int = 6,
) -> None:
    data = measured.copy()
    data = data[
        np.isfinite(data["recompute_ms"])
        & np.isfinite(data["tensor_mb"])
        & (data["recompute_ms"] > 0)
        & (data["tensor_mb"] >= 0)
    ].copy()

    # Communication saved per millisecond of recomputation.
    data["mb_per_recompute_ms"] = (
        data["tensor_mb"] / data["recompute_ms"]
    )

    # Shorter labels are much easier to read than the full JIN keys.
    data["short_label"] = data["target"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Point size represents tensor size as an additional visual cue.
    sizes = 45 + 28 * data["tensor_mb"].clip(lower=0)

    ax.scatter(
        data["recompute_ms"],
        data["tensor_mb"],
        s=sizes,
        alpha=0.72,
        edgecolors="black",
        linewidths=0.4,
    )

    # The low-cost candidates are packed close together, so a log x-axis
    # spreads them out and makes the left side much easier to inspect.
    ax.set_xscale("log")

    # Annotate only the best candidates, and stagger the labels vertically.
    top = (
        data.sort_values(
            ["mb_per_recompute_ms", "tensor_mb"],
            ascending=[False, False],
        )
        .drop_duplicates(subset=["short_label"])
        .head(annotate_top)
        .reset_index(drop=True)
    )

    offsets = [
        (8, 12),
        (8, -18),
        (8, 24),
        (8, -30),
        (8, 36),
        (8, -42),
    ]

    for i, row in top.iterrows():
        offset = offsets[i % len(offsets)]
        ax.annotate(
            row["short_label"],
            (row["recompute_ms"], row["tensor_mb"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.6,
                "alpha": 0.65,
            },
        )

    ax.set_xlabel("Measured recompute cost (ms, log scale)")
    ax.set_ylabel("Saved tensor size (MB)")
    ax.set_title("Recompute candidate efficiency")
    ax.grid(True, which="both", alpha=0.25)

    # Explain how to read the plot.
    ax.text(
        0.02,
        0.98,
        "Better candidates are toward the upper-left",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_e2e_summary(e2e: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "payload_mb",
        "estimated_grouped_ms",
        "recompute_wall_ms",
        "inject_ms",
        "torch_backward_ms",
        "request_round_trip_ms",
        "node_b_processing_ms",
        "node_a_total_ms",
    ]

    grouped = (
        e2e.groupby(
            ["network_limit_mbps", "drop_ratio"],
            as_index=False,
        )[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten the MultiIndex columns.
    flat_columns: list[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            left, right = col
            flat_columns.append(left if right == "" else f"{left}_{right}")
        else:
            flat_columns.append(str(col))
    grouped.columns = flat_columns

    grouped["network_label"] = grouped["network_limit_mbps"].map(network_label)
    return grouped.sort_values(
        ["network_limit_mbps", "drop_ratio"]
    ).reset_index(drop=True)


def plot_total_runtime(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for network, group in summary.groupby(
        "network_limit_mbps",
        sort=True,
    ):
        group = group.sort_values("drop_ratio")
        ax.errorbar(
            group["drop_ratio"],
            group["node_a_total_ms_mean"],
            yerr=group["node_a_total_ms_std"].fillna(0),
            marker="o",
            capsize=3,
            label=network_label(network),
        )

    ax.set_xlabel("Drop ratio")
    ax.set_ylabel("Node A total runtime (ms)")
    ax.set_title("Total runtime vs. drop ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Network")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_runtime_breakdown(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    for network, group in summary.groupby(
        "network_limit_mbps",
        sort=True,
    ):
        group = group.sort_values("drop_ratio").copy()

        # Communication is represented by request round trip.
        communication = group["request_round_trip_ms_mean"]

        # Recompute pipeline includes recomputation and reinjection.
        recompute_pipeline = (
            group["recompute_wall_ms_mean"]
            + group["inject_ms_mean"]
        )

        # Residual category so stacked components sum to total runtime.
        other = (
            group["node_a_total_ms_mean"]
            - communication
            - recompute_pipeline
        ).clip(lower=0)

        x = np.arange(len(group))
        width = 0.72

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.bar(
            x,
            other,
            width,
            label="Other processing",
        )
        ax.bar(
            x,
            recompute_pipeline,
            width,
            bottom=other,
            label="Recompute pipeline",
        )
        ax.bar(
            x,
            communication,
            width,
            bottom=other + recompute_pipeline,
            label="Communication",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in group["drop_ratio"]])
        ax.set_xlabel("Drop ratio")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Runtime breakdown — {network_label(network)}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        fig.tight_layout()

        suffix = (
            "unlimited"
            if float(network) == 0.0
            else f"{float(network):g}_mbps"
        )
        fig.savefig(
            output_dir / f"04_runtime_breakdown_{suffix}.png",
            dpi=220,
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer-profile",
        type=Path,
        default=Path("recompute_layer_profile.csv"),
    )
    parser.add_argument(
        "--estimated-menu",
        type=Path,
        default=Path("recompute_menu.csv"),
    )
    parser.add_argument(
        "--measured-menu",
        type=Path,
        default=Path("recompute_menu_resnet18_bs32_c1.csv"),
    )
    parser.add_argument(
        "--e2e",
        type=Path,
        default=Path("drop_ratio_experiment_all_networks.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
    )
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, estimated, measured, e2e = load_and_validate(
        args.layer_profile,
        args.estimated_menu,
        args.measured_menu,
        args.e2e,
    )

    validation = build_validation_table(estimated, measured)
    validation.to_csv(
        args.output_dir / "cost_model_validation.csv",
        index=False,
    )
    print_validation_metrics(validation)

    summary = build_e2e_summary(e2e)
    summary.to_csv(
        args.output_dir / "end_to_end_summary.csv",
        index=False,
    )

    plot_predicted_vs_measured(
        validation,
        args.output_dir / "01_predicted_vs_measured.png",
    )
    plot_candidate_efficiency(
        measured,
        args.output_dir / "02_candidate_efficiency.png",
        annotate_top=args.annotate_top,
    )
    plot_total_runtime(
        summary,
        args.output_dir / "03_total_runtime_vs_drop_ratio.png",
    )
    plot_runtime_breakdown(summary, args.output_dir)

    print(f"\n[DONE] Outputs saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
