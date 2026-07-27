import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


CSV_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "recompute_cost_ratio_500mbps_6runs.csv"
)


# CSV 읽기
df = pd.read_csv(CSV_PATH)

# 실제 recompute를 수행한 행만 사용
# drop_ratio == 0은 모든 recompute 값이 0인 baseline이므로 제외
df = df[df["drop_ratio"] > 0].copy()


TARGET = "total_recompute_cost_ms"

MODELS = {
    "Operator only": [
        "predicted_operator_ms",
    ],

    "Recomputed MB only": [
        "recomputed_mb",
    ],

    "Node count only": [
        "recompute_executed_node_count",
    ],

    "Operator + MB": [
        "predicted_operator_ms",
        "recomputed_mb",
    ],

    "Operator + MB + Nodes": [
        "predicted_operator_ms",
        "recomputed_mb",
        "recompute_executed_node_count",
    ],
}


results = []

for model_name, features in MODELS.items():

    X = df[features]
    y = df[TARGET]

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(X)

    r2 = r2_score(y, prediction)
    rmse = np.sqrt(
        mean_squared_error(y, prediction)
    )
    mae = mean_absolute_error(y, prediction)

    equation_parts = [
        f"{model.intercept_:.4f}"
    ]

    for feature, coefficient in zip(
        features,
        model.coef_,
    ):
        equation_parts.append(
            f"({coefficient:.4f} × {feature})"
        )

    equation = " + ".join(equation_parts)

    results.append(
        {
            "model": model_name,
            "features": ", ".join(features),
            "r2": r2,
            "rmse_ms": rmse,
            "mae_ms": mae,
            "equation": equation,
        }
    )


results_df = pd.DataFrame(results)

# R²가 높은 순서로 정렬
results_df = results_df.sort_values(
    by="r2",
    ascending=False,
).reset_index(drop=True)


print()
print("Recompute Cost Model Comparison")
print("=" * 100)

print(
    results_df[
        [
            "model",
            "r2",
            "rmse_ms",
            "mae_ms",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("Equations")
print("=" * 100)

for _, row in results_df.iterrows():
    print()
    print(f"[{row['model']}]")
    print(
        "total_recompute_cost_ms = "
        + row["equation"]
    )


output_path = "analysis/model_comparison.csv"

results_df.to_csv(
    output_path,
    index=False,
)

print()
print(f"[SAVE] {output_path}")