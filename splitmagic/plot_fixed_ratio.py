
# python3 plot_fixed_ratio.py \
#     --results fixed_ratio_sweep_all_networks.csv \
#     --network-stats fixed_ratio_sweep_all_networks_tc.csv \
#     --output-dir figures_fixed_ratio
#!/usr/bin/env python3
"""Generate figures and oracle summary for fixed-ratio sweep experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

EXPECTED_BANDWIDTH_ORDER = [1000, 500, 200, 100, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("fixed_ratio_sweep_all_networks.csv"),
    )
    parser.add_argument(
        "--network-stats",
        type=Path,
        default=Path("fixed_ratio_sweep_all_networks_tc.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures_fixed_ratio"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def load_and_merge(results_path: Path, network_stats_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    results = pd.read_csv(results_path)
    require_columns(results, ["run_id", "drop_ratio", "node_a_total_ms"], results_path.name)

    possible_bw_cols = ["network_limit_mbps", "network_mbps", "bandwidth_mbps"]
    bw_col = next((c for c in possible_bw_cols if c in results.columns), None)

    if bw_col is not None:
        merged = results.rename(columns={bw_col: "bandwidth_mbps"}).copy()
    else:
        if not network_stats_path.exists():
            raise FileNotFoundError(
                "Main CSV has no bandwidth column and network stats CSV was not found: "
                f"{network_stats_path}"
            )

        network = pd.read_csv(network_stats_path)
        require_columns(network, ["run_id", "network_limit_mbps"], network_stats_path.name)

        keep = [c for c in [
            "run_id", "repeat", "hostname", "selection_policy",
            "configured_drop_ratio", "network_limit_mbps", "tc_bytes_delta"
        ] if c in network.columns]

        network = network[keep].rename(columns={"network_limit_mbps": "bandwidth_mbps"})
        merged = results.merge(network, on="run_id", how="left", validate="one_to_one")

    if merged["bandwidth_mbps"].isna().any():
        bad = merged.loc[merged["bandwidth_mbps"].isna(), "run_id"].tolist()
        raise ValueError(f"Could not recover bandwidth for run_id values: {bad}")

    numeric_cols = [
        "run_id", "drop_ratio", "bandwidth_mbps", "node_a_total_ms",
        "payload_mb", "saved_mb", "recompute_wall_ms", "recompute_overhead_ms",
        "predicted_operator_ms", "estimated_grouped_ms", "request_round_trip_ms",
        "node_b_processing_ms", "backward_jin_ms", "inject_ms",
        "torch_backward_ms", "tc_bytes_delta"
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["bandwidth_mbps"] = merged["bandwidth_mbps"].astype(int)
    return merged.sort_values(
        ["bandwidth_mbps", "drop_ratio", "run_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def bandwidth_order(df: pd.DataFrame) -> list[int]:
    available = set(df["bandwidth_mbps"].dropna().astype(int).unique())
    ordered = [bw for bw in EXPECTED_BANDWIDTH_ORDER if bw in available]
    return ordered + sorted(available.difference(ordered), reverse=True)


def aggregate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").columns.tolist()
    excluded = {"run_id", "bandwidth_mbps", "drop_ratio"}
    values = [c for c in numeric if c not in excluded]

    grouped = df.groupby(["bandwidth_mbps", "drop_ratio"], as_index=False)[values].mean()
    counts = (
        df.groupby(["bandwidth_mbps", "drop_ratio"])
        .size().rename("num_repeats").reset_index()
    )
    return grouped.merge(counts, on=["bandwidth_mbps", "drop_ratio"], how="left")


def finish(path: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                output: Path, dpi: int, highlight_min: bool = False) -> None:
    if metric not in df.columns:
        print(f"[SKIP] Missing column: {metric}")
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for bw in bandwidth_order(df):
        sub = df[df["bandwidth_mbps"] == bw].sort_values("drop_ratio")
        ax.plot(sub["drop_ratio"], sub[metric], marker="o", linewidth=1.8,
                markersize=4.8, label=f"{bw} Mbps")

        if highlight_min and sub[metric].notna().any():
            best = sub.loc[sub[metric].idxmin()]
            ax.scatter([best["drop_ratio"]], [best[metric]], s=85,
                       facecolors="none", edgecolors="black", linewidths=1.2,
                       zorder=5)

    ax.set_xlabel("Activation Drop Ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(df["drop_ratio"].dropna().unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Bandwidth")
    finish(output, dpi)


def plot_tradeoff(df: pd.DataFrame, output: Path, dpi: int) -> None:
    required = ["request_round_trip_ms", "recompute_wall_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[SKIP] Tradeoff plot missing: {', '.join(missing)}")
        return

    bws = bandwidth_order(df)
    fig, axes = plt.subplots(len(bws), 1,
                             figsize=(8.3, max(3.2 * len(bws), 5.0)),
                             sharex=True)
    if len(bws) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bws):
        sub = df[df["bandwidth_mbps"] == bw].sort_values("drop_ratio")
        ax.plot(sub["drop_ratio"], sub["request_round_trip_ms"],
                marker="o", label="Request round trip")
        ax.plot(sub["drop_ratio"], sub["recompute_wall_ms"],
                marker="s", label="Recompute wall")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{bw} Mbps")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Activation Drop Ratio")
    axes[-1].set_xticks(sorted(df["drop_ratio"].dropna().unique()))
    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle("Communication–Recomputation Tradeoff by Bandwidth", y=1.002)
    finish(output, dpi)


def build_oracle(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bw in bandwidth_order(df):
        sub = df[df["bandwidth_mbps"] == bw].dropna(subset=["node_a_total_ms"])
        if sub.empty:
            continue
        best = sub.loc[sub["node_a_total_ms"].idxmin()]
        row = {
            "bandwidth_mbps": int(bw),
            "best_drop_ratio": float(best["drop_ratio"]),
            "best_node_a_total_ms": float(best["node_a_total_ms"]),
        }
        for col in [
            "payload_mb", "saved_mb", "recompute_wall_ms",
            "request_round_trip_ms", "node_b_processing_ms", "backward_jin_ms",
            "predicted_operator_ms", "estimated_grouped_ms", "tc_bytes_delta",
            "num_repeats"
        ]:
            if col in best.index and pd.notna(best[col]):
                row[col] = float(best[col])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_oracle_ratio(oracle: pd.DataFrame, output: Path, dpi: int) -> None:
    if oracle.empty:
        return
    ordered = oracle.sort_values("bandwidth_mbps", ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(ordered["bandwidth_mbps"], ordered["best_drop_ratio"],
            marker="o", linewidth=1.8)
    for _, row in ordered.iterrows():
        ax.annotate(f"{row['best_drop_ratio']:.2f}",
                    (row["bandwidth_mbps"], row["best_drop_ratio"]),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(ordered["bandwidth_mbps"])
    ax.set_xticklabels([str(v) for v in ordered["bandwidth_mbps"]])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Oracle Drop Ratio")
    ax.set_title("Oracle Fixed Drop Ratio by Bandwidth")
    ax.grid(True, alpha=0.3)
    finish(output, dpi)


def plot_oracle_runtime(oracle: pd.DataFrame, output: Path, dpi: int) -> None:
    if oracle.empty:
        return
    ordered = oracle.sort_values("bandwidth_mbps", ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(ordered["bandwidth_mbps"], ordered["best_node_a_total_ms"],
            marker="o", linewidth=1.8)
    for _, row in ordered.iterrows():
        ax.annotate(f"{row['best_node_a_total_ms']:.0f}",
                    (row["bandwidth_mbps"], row["best_node_a_total_ms"]),
                    textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(ordered["bandwidth_mbps"])
    ax.set_xticklabels([str(v) for v in ordered["bandwidth_mbps"]])
    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Best Node A Total Time (ms)")
    ax.set_title("Oracle Runtime by Bandwidth")
    ax.grid(True, alpha=0.3)
    finish(output, dpi)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_merge(args.results, args.network_stats)
    aggregated = aggregate_repeats(merged)

    merged_path = args.output_dir / "merged_fixed_ratio_results.csv"
    aggregated.to_csv(merged_path, index=False)

    oracle = build_oracle(aggregated)
    oracle_path = args.output_dir / "oracle_summary.csv"
    oracle.to_csv(oracle_path, index=False)

    plot_metric(aggregated, "node_a_total_ms", "Node A Total Time (ms)",
                "Fixed-Ratio Runtime Across Bandwidths",
                args.output_dir / "figure1_runtime_vs_drop_ratio.png",
                args.dpi, highlight_min=True)
    plot_metric(aggregated, "payload_mb", "Payload Size (MB)",
                "Payload Size vs Activation Drop Ratio",
                args.output_dir / "figure2_payload_vs_drop_ratio.png", args.dpi)
    plot_metric(aggregated, "recompute_wall_ms", "Recompute Wall Time (ms)",
                "Recomputation Cost vs Activation Drop Ratio",
                args.output_dir / "figure3_recompute_vs_drop_ratio.png", args.dpi)
    plot_metric(aggregated, "request_round_trip_ms", "Request Round-Trip Time (ms)",
                "Request Round-Trip Time vs Activation Drop Ratio",
                args.output_dir / "figure4_round_trip_vs_drop_ratio.png", args.dpi)
    plot_tradeoff(aggregated,
                  args.output_dir / "figure5_tradeoff_by_bandwidth.png", args.dpi)
    plot_oracle_ratio(oracle,
                      args.output_dir / "figure6_oracle_ratio_vs_bandwidth.png", args.dpi)
    plot_oracle_runtime(oracle,
                        args.output_dir / "figure7_oracle_runtime_vs_bandwidth.png", args.dpi)

    print("\n[OUTPUT]")
    print(f"  directory  : {args.output_dir}")
    print(f"  merged CSV : {merged_path}")
    print(f"  oracle CSV : {oracle_path}")
    if not oracle.empty:
        print("\n[ORACLE SUMMARY]")
        print(oracle.to_string(index=False))
    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)