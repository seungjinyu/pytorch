import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# 사용법:
# python3 analyze_recompute_cost.py
# 또는
# python3 analyze_recompute_cost.py recompute_cost_ratio_500mbps_6runs.csv

CSV_PATH = (
    sys.argv[1]
    if len(sys.argv) >= 2
    else "recompute_cost_ratio_500mbps_6runs.csv"
)

OUTDIR = "analysis"
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)

required_columns = [
    "drop_ratio",
    "predicted_operator_ms",
    "recompute_wall_ms",
    "recompute_overhead_ms",
    "recomputed_mb",
    "inject_ms",
    "total_recompute_cost_ms",
    "recompute_executed_node_count",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "CSV에 필요한 컬럼이 없습니다:\n"
        + "\n".join(f"  - {column}" for column in missing_columns)
        + "\n\n현재 컬럼:\n"
        + "\n".join(f"  - {column}" for column in df.columns)
    )


print(f"[LOAD] {CSV_PATH}")
print(f"[LOAD] rows={len(df)}, columns={len(df.columns)}")


# ---------------------------------------------------------
# Summary statistics by drop ratio
# ---------------------------------------------------------

summary = (
    df.groupby("drop_ratio", as_index=False)
    .agg(
        run_count=("run_id", "count"),

        payload_mb_mean=("payload_mb", "mean"),

        saved_mb_mean=("saved_mb", "mean"),
        saved_mb_std=("saved_mb", "std"),

        predicted_operator_mean=(
            "predicted_operator_ms",
            "mean",
        ),
        predicted_operator_std=(
            "predicted_operator_ms",
            "std",
        ),

        recompute_wall_mean=(
            "recompute_wall_ms",
            "mean",
        ),
        recompute_wall_std=(
            "recompute_wall_ms",
            "std",
        ),

        recompute_overhead_mean=(
            "recompute_overhead_ms",
            "mean",
        ),
        recompute_overhead_std=(
            "recompute_overhead_ms",
            "std",
        ),

        inject_mean=("inject_ms", "mean"),
        inject_std=("inject_ms", "std"),

        total_cost_mean=(
            "total_recompute_cost_ms",
            "mean",
        ),
        total_cost_std=(
            "total_recompute_cost_ms",
            "std",
        ),

        recomputed_mb_mean=(
            "recomputed_mb",
            "mean",
        ),
        recomputed_mb_std=(
            "recomputed_mb",
            "std",
        ),

        executed_nodes_mean=(
            "recompute_executed_node_count",
            "mean",
        ),
        executed_nodes_std=(
            "recompute_executed_node_count",
            "std",
        ),
    )
)

summary_path = os.path.join(OUTDIR, "summary.csv")
summary.to_csv(summary_path, index=False)

print(f"[SAVE] {summary_path}")


# ---------------------------------------------------------
# Figure 1: Total recompute cost by drop ratio
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    summary["drop_ratio"],
    summary["total_cost_mean"],
    yerr=summary["total_cost_std"].fillna(0),
    marker="o",
    capsize=4,
)

plt.xlabel("Drop ratio")
plt.ylabel("Total recompute cost (ms)")
plt.title("Total Recompute Cost vs. Drop Ratio")
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = os.path.join(
    OUTDIR,
    "figure1_total_cost_vs_drop_ratio.png",
)

plt.savefig(path, dpi=300)
plt.close()

print(f"[SAVE] {path}")


# ---------------------------------------------------------
# Figure 2: Recompute cost components
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    summary["drop_ratio"],
    summary["recompute_wall_mean"],
    marker="o",
    label="Recompute wall time",
)

plt.plot(
    summary["drop_ratio"],
    summary["inject_mean"],
    marker="s",
    label="Inject time",
)

plt.plot(
    summary["drop_ratio"],
    summary["total_cost_mean"],
    marker="^",
    label="Total recompute cost",
)

plt.xlabel("Drop ratio")
plt.ylabel("Time (ms)")
plt.title("Recompute Cost Components")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(
    OUTDIR,
    "figure2_cost_components.png",
)

plt.savefig(path, dpi=300)
plt.close()

print(f"[SAVE] {path}")


# ---------------------------------------------------------
# Figure 3: Predicted operator cost vs recompute wall time
# ---------------------------------------------------------

positive_df = df[df["drop_ratio"] > 0].copy()

plt.figure(figsize=(6, 5))

plt.scatter(
    positive_df["predicted_operator_ms"],
    positive_df["recompute_wall_ms"],
)

plt.xlabel("Predicted operator time (ms)")
plt.ylabel("Measured recompute wall time (ms)")
plt.title("Operator Prediction vs. Recompute Wall Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = os.path.join(
    OUTDIR,
    "figure3_operator_vs_wall.png",
)

plt.savefig(path, dpi=300)
plt.close()

print(f"[SAVE] {path}")


# ---------------------------------------------------------
# Figure 4: Recomputed MB vs inject time
# ---------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.scatter(
    positive_df["recomputed_mb"],
    positive_df["inject_ms"],
)

inject_model = LinearRegression()

inject_X = positive_df[["recomputed_mb"]]
inject_y = positive_df["inject_ms"]

inject_model.fit(inject_X, inject_y)

inject_prediction = inject_model.predict(inject_X)

sort_indices = np.argsort(
    positive_df["recomputed_mb"].to_numpy()
)

plt.plot(
    positive_df["recomputed_mb"].to_numpy()[sort_indices],
    inject_prediction[sort_indices],
)

inject_r2 = r2_score(
    inject_y,
    inject_prediction,
)

plt.xlabel("Recomputed data (MB)")
plt.ylabel("Inject time (ms)")
plt.title(
    f"Inject Time vs. Recomputed Data\n"
    f"$R^2$ = {inject_r2:.4f}"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = os.path.join(
    OUTDIR,
    "figure4_inject_vs_recomputed_mb.png",
)

plt.savefig(path, dpi=300)
plt.close()

print(f"[SAVE] {path}")


# ---------------------------------------------------------
# Regression for total recompute cost
#
# 0.0 rows are excluded because they represent no-recompute
# baseline rows rather than an executed recomputation.
# ---------------------------------------------------------

features = [
    "predicted_operator_ms",
    "recomputed_mb",
    "recompute_executed_node_count",
]

regression_df = df[df["drop_ratio"] > 0].copy()

X = regression_df[features]
y = regression_df["total_recompute_cost_ms"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(X)

r2 = r2_score(y, prediction)
rmse = np.sqrt(
    mean_squared_error(y, prediction)
)
mae = mean_absolute_error(y, prediction)


# ---------------------------------------------------------
# Save regression results
# ---------------------------------------------------------

regression_path = os.path.join(
    OUTDIR,
    "regression.txt",
)

with open(regression_path, "w") as file:
    file.write(
        "Total Recompute Cost Linear Regression\n"
    )
    file.write("=" * 45 + "\n\n")

    file.write(
        "Target:\n"
        "  total_recompute_cost_ms\n\n"
    )

    file.write("Features:\n")

    for feature, coefficient in zip(
        features,
        model.coef_,
    ):
        file.write(
            f"  {feature}: "
            f"{coefficient:.8f}\n"
        )

    file.write(
        f"\nIntercept: "
        f"{model.intercept_:.8f}\n"
    )

    file.write("\nMetrics:\n")
    file.write(f"  R2:   {r2:.6f}\n")
    file.write(f"  RMSE: {rmse:.6f} ms\n")
    file.write(f"  MAE:  {mae:.6f} ms\n")

    file.write("\nEquation:\n")
    file.write(
        "  total_recompute_cost_ms =\n"
        f"      {model.intercept_:.8f}\n"
    )

    for feature, coefficient in zip(
        features,
        model.coef_,
    ):
        file.write(
            f"    + ({coefficient:.8f})"
            f" * {feature}\n"
        )

print(f"[SAVE] {regression_path}")


# ---------------------------------------------------------
# Figure 5: Regression predicted vs actual
# ---------------------------------------------------------

plt.figure(figsize=(6, 6))

plt.scatter(
    y,
    prediction,
)

minimum = min(
    float(y.min()),
    float(prediction.min()),
)

maximum = max(
    float(y.max()),
    float(prediction.max()),
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
)

plt.xlabel("Actual total recompute cost (ms)")
plt.ylabel("Regression-predicted cost (ms)")
plt.title(
    f"Predicted vs. Actual Recompute Cost\n"
    f"$R^2$ = {r2:.4f}, "
    f"RMSE = {rmse:.2f} ms"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = os.path.join(
    OUTDIR,
    "figure5_regression_prediction.png",
)

plt.savefig(path, dpi=300)
plt.close()

print(f"[SAVE] {path}")


# ---------------------------------------------------------
# Figure 6: Residuals
# ---------------------------------------------------------

residuals = y - prediction

plt.figure(figsize=(7, 4))

plt.scatter(
    prediction,
    residuals,
)

plt.axhline(
    y=0,
    linestyle="--",
)

plt.xlabel("Predicted total recompute cost (ms)")
plt.ylabel("Residual: actual - predicted (ms)")
plt.title("Regression Residuals")
plt.grid(True, alpha=0.3)
plt.tight_layout()

path = os.path.join(
    OUTDIR,
    "figure6_regression_residuals.png",
)

plt.savefig(path, dpi=300)
plt.close()

print(f"[SAVE] {path}")


# ---------------------------------------------------------
# Terminal output
# ---------------------------------------------------------

display_columns = [
    "drop_ratio",
    "run_count",
    "saved_mb_mean",
    "predicted_operator_mean",
    "recompute_wall_mean",
    "inject_mean",
    "total_cost_mean",
    "total_cost_std",
    "recomputed_mb_mean",
    "executed_nodes_mean",
]

print()
print("Summary")
print("=" * 120)

print(
    summary[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print()
print("Regression")
print("=" * 50)
print(f"Intercept = {model.intercept_:.6f}")

for feature, coefficient in zip(
    features,
    model.coef_,
):
    print(
        f"{feature:35s} "
        f"{coefficient:12.6f}"
    )

print()
print(f"R2   = {r2:.6f}")
print(f"RMSE = {rmse:.6f} ms")
print(f"MAE  = {mae:.6f} ms")
print(f"Inject R2 = {inject_r2:.6f}")

print()
print("=" * 60)
print("Feature Correlation")
print("=" * 60)

print(
    df[
        [
            "predicted_operator_ms",
            "recomputed_mb",
            "recompute_executed_node_count",
        ]
    ].corr()
)