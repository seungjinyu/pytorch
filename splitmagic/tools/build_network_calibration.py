import argparse
import json

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    x = df["payload_mb"].to_numpy(dtype=float)
    y = df["request_round_trip_ms"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    result = {
        "send_fixed_ms": float(intercept),
        "send_ms_per_mb": float(slope),
        "num_rows": int(len(df)),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(result)


if __name__ == "__main__":
    main()