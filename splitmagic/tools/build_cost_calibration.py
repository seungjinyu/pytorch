import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Calibration experiment CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output calibration JSON",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {
        "predicted_operator_ms",
        "recompute_wall_ms",
        "recompute_overhead_ms",
        "recomputed_mb",
        "inject_ms",
        "missing_count",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    valid_operator = df[
        df["predicted_operator_ms"] > 0
    ].copy()

    if valid_operator.empty:
        raise ValueError(
            "No valid rows with predicted_operator_ms > 0"
        )

    x = valid_operator["predicted_operator_ms"].to_numpy()
    y = valid_operator["recompute_wall_ms"].to_numpy()

    operator_scale, recompute_fixed_ms = np.polyfit(
        x,
        y,
        1,
    )

    valid_inject = df[
        df["recomputed_mb"] > 0
    ].copy()

    if valid_inject.empty:
        inject_ms_per_mb = 0.0
    else:
        inject_ms_per_mb = float(
            (
                valid_inject["inject_ms"]
                / valid_inject["recomputed_mb"]
            ).median()
        )

    calibration = {
        "version": 1,
        "device": args.device,
        "batch_size": args.batch_size,

        "operator_scale": float(operator_scale),
        "recompute_fixed_ms": float(recompute_fixed_ms),

        "inject_ms_per_mb": inject_ms_per_mb,

        "num_rows": int(len(df)),
    }

    output_path = Path(args.output)

    with output_path.open("w") as f:
        json.dump(
            calibration,
            f,
            indent=2,
        )

    print(
        json.dumps(
            calibration,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()