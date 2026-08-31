#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys

PROFILE_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tests",
    "test_operator_concurrency_resnet18_mp.py",
)

NODE_LIST_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tests",
    "test_operator_concurrency_resnet18.py",
)


def get_nodes(device, dataset, batch_size):
    cmd = [
        sys.executable,
        NODE_LIST_SCRIPT,
        "--device", device,
        "--dataset", dataset,
        "--batch-size", str(batch_size),
        "--list",
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    nodes = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line or "name=" not in line:
            continue

        # example:
        #  1 name=conv1 op=call_module target=conv1

        parts = line.split()

        for part in parts:
            if part.startswith("name="):
                node_name = part.split("=", 1)[1]
                nodes.append(node_name)
                break

    return nodes


def run_one(
    device,
    dataset,
    batch_size,
    node,
    concurrency,
    repeats,
):
    cmd = [
        sys.executable,
        PROFILE_SCRIPT,
        "--device", device,
        "--dataset", dataset,
        "--batch-size", str(batch_size),
        "--node", node,
        "--concurrency", str(concurrency),
        "--repeats", str(repeats),
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = {}

    wanted = {
        "node_name",
        "device",
        "concurrency",
        "workers",
        "repeats_per_worker",
        "samples",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
    }

    for line in result.stdout.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key not in wanted:
            continue

        parsed[key] = value

    if "node_name" not in parsed:
        raise RuntimeError(
            f"Could not parse result for node={node}, "
            f"device={device}, concurrency={concurrency}"
        )

    return parsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["cifar10", "imagenet"],
        default="cifar10",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"],
    )

    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=[1, 2, 4, 6, 8],
    )

    parser.add_argument(
        "--output-dir",
        default="./operator_concurrency_results",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("============================================================")
    print("ResNet18 Operator Concurrency Sweep")
    print("============================================================")
    print(f"dataset       = {args.dataset}")
    print(f"batch_size    = {args.batch_size}")
    print(f"repeats       = {args.repeats}")
    print(f"devices       = {args.devices}")
    print(f"concurrencies = {args.concurrencies}")
    print(f"output_dir    = {args.output_dir}")
    print("============================================================")

    for device in args.devices:

        print()
        print("============================================================")
        print(f"[DEVICE] {device}")
        print("============================================================")

        nodes = get_nodes(
            device=device,
            dataset=args.dataset,
            batch_size=args.batch_size,
        )

        print(f"[INFO] nodes={len(nodes)}")

        rows = []

        total = len(nodes) * len(args.concurrencies)
        current = 0

        for node in nodes:
            for concurrency in args.concurrencies:
                current += 1

                print(
                    f"[RUN {current}/{total}] "
                    f"device={device} "
                    f"node={node} "
                    f"concurrency={concurrency}"
                )

                try:
                    row = run_one(
                        device=device,
                        dataset=args.dataset,
                        batch_size=args.batch_size,
                        node=node,
                        concurrency=concurrency,
                        repeats=args.repeats,
                    )

                    rows.append(row)

                    print(
                        f"[OK] "
                        f"median={row['median_ms']} ms "
                        f"p95={row['p95_ms']} ms"
                    )

                except subprocess.CalledProcessError as e:
                    print(
                        f"[FAIL] node={node} "
                        f"concurrency={concurrency}"
                    )

                    print(e.stdout)
                    print(e.stderr)

                except Exception as e:
                    print(
                        f"[FAIL] node={node} "
                        f"concurrency={concurrency} "
                        f"error={e}"
                    )

        output_csv = os.path.join(
            args.output_dir,
            f"resnet18_{args.dataset}_{device}_"
            f"concurrency_bs{args.batch_size}.csv"
        )

        fieldnames = [
            "node_name",
            "device",
            "concurrency",
            "workers",
            "repeats_per_worker",
            "samples",
            "median_ms",
            "mean_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
        ]

        with open(
            output_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        print()
        print(f"[DONE] {device}")
        print(f"[CSV] {output_csv}")


if __name__ == "__main__":
    main()