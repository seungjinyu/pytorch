#!/usr/bin/env python3
"""
Unified NVTX/Nsight operator concurrency sweep.

Replaces:
  run_op_con_sweep_resnet18.py
  run_op_con_sweep_resnet50.py
  run_op_con_sweep_tinystories.py

Behavior
--------
- ResNet CPU: existing non-NVTX MP test.
- ResNet CUDA: existing NVTX MP test wrapped by `nsys profile`.
- TinyStories CUDA: uses test_op_con_nvtx_tinystories_mp.py if it exists.
  If that file is absent, it falls back to the existing regular TinyStories
  MP profiler and prints a warning.
- TinyStories CPU: regular TinyStories MP profiler.

This preserves the current repository behavior while giving one CLI.
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
TEST_DIR = REPO_ROOT / "tests"

TINYSTORIES_MODEL_NAME = "roneneldan/TinyStories-33M"
TINYSTORIES_PARAMETERS = 68_514_048
TINYSTORIES_DEFAULT_NODES = ['transformer_wte', 'transformer_wpe', 'transformer_h_0_ln_1', 'transformer_h_0_attn_attention_q_proj', 'transformer_h_0_attn_attention_k_proj', 'transformer_h_0_attn_attention_v_proj', 'matmul', 'softmax', 'matmul_1', 'transformer_h_0_attn_attention_out_proj', 'add_10', 'transformer_h_0_ln_2', 'transformer_h_0_mlp_c_fc', 'block0_gelu', 'transformer_h_0_mlp_c_proj', 'add_13', 'transformer_h_1_ln_1', 'transformer_h_1_attn_attention_q_proj', 'transformer_h_1_attn_attention_k_proj', 'transformer_h_1_attn_attention_v_proj', 'matmul_2', 'softmax_1', 'matmul_3', 'transformer_h_1_attn_attention_out_proj', 'add_19', 'transformer_h_1_ln_2', 'transformer_h_1_mlp_c_fc', 'block1_gelu', 'transformer_h_1_mlp_c_proj', 'add_22', 'transformer_h_2_ln_1', 'transformer_h_2_attn_attention_q_proj', 'transformer_h_2_attn_attention_k_proj', 'transformer_h_2_attn_attention_v_proj', 'matmul_4', 'softmax_2', 'matmul_5', 'transformer_h_2_attn_attention_out_proj', 'add_28', 'transformer_h_2_ln_2', 'transformer_h_2_mlp_c_fc', 'block2_gelu', 'transformer_h_2_mlp_c_proj', 'add_31', 'transformer_h_3_ln_1', 'transformer_h_3_attn_attention_q_proj', 'transformer_h_3_attn_attention_k_proj', 'transformer_h_3_attn_attention_v_proj', 'matmul_6', 'softmax_3', 'matmul_7', 'transformer_h_3_attn_attention_out_proj', 'add_37', 'transformer_h_3_ln_2', 'transformer_h_3_mlp_c_fc', 'block3_gelu', 'transformer_h_3_mlp_c_proj', 'add_40', 'transformer_ln_f', 'lm_head']
TINYSTORIES_SHORT_NAMES = {'transformer_wte': 'embedding_token', 'transformer_wpe': 'embedding_position', 'transformer_h_0_ln_1': 'block0_ln1', 'transformer_h_0_attn_attention_q_proj': 'block0_q_proj', 'transformer_h_0_attn_attention_k_proj': 'block0_k_proj', 'transformer_h_0_attn_attention_v_proj': 'block0_v_proj', 'matmul': 'block0_attn_qk_matmul', 'softmax': 'block0_attn_softmax', 'matmul_1': 'block0_attn_v_matmul', 'transformer_h_0_attn_attention_out_proj': 'block0_out_proj', 'add_10': 'block0_residual_attn', 'transformer_h_0_ln_2': 'block0_ln2', 'transformer_h_0_mlp_c_fc': 'block0_mlp_c_fc', 'block0_gelu': 'block0_gelu', 'transformer_h_0_mlp_c_proj': 'block0_mlp_c_proj', 'add_13': 'block0_residual_mlp', 'transformer_h_1_ln_1': 'block1_ln1', 'transformer_h_1_attn_attention_q_proj': 'block1_q_proj', 'transformer_h_1_attn_attention_k_proj': 'block1_k_proj', 'transformer_h_1_attn_attention_v_proj': 'block1_v_proj', 'matmul_2': 'block1_attn_qk_matmul', 'softmax_1': 'block1_attn_softmax', 'matmul_3': 'block1_attn_v_matmul', 'transformer_h_1_attn_attention_out_proj': 'block1_out_proj', 'add_19': 'block1_residual_attn', 'transformer_h_1_ln_2': 'block1_ln2', 'transformer_h_1_mlp_c_fc': 'block1_mlp_c_fc', 'block1_gelu': 'block1_gelu', 'transformer_h_1_mlp_c_proj': 'block1_mlp_c_proj', 'add_22': 'block1_residual_mlp', 'transformer_h_2_ln_1': 'block2_ln1', 'transformer_h_2_attn_attention_q_proj': 'block2_q_proj', 'transformer_h_2_attn_attention_k_proj': 'block2_k_proj', 'transformer_h_2_attn_attention_v_proj': 'block2_v_proj', 'matmul_4': 'block2_attn_qk_matmul', 'softmax_2': 'block2_attn_softmax', 'matmul_5': 'block2_attn_v_matmul', 'transformer_h_2_attn_attention_out_proj': 'block2_out_proj', 'add_28': 'block2_residual_attn', 'transformer_h_2_ln_2': 'block2_ln2', 'transformer_h_2_mlp_c_fc': 'block2_mlp_c_fc', 'block2_gelu': 'block2_gelu', 'transformer_h_2_mlp_c_proj': 'block2_mlp_c_proj', 'add_31': 'block2_residual_mlp', 'transformer_h_3_ln_1': 'block3_ln1', 'transformer_h_3_attn_attention_q_proj': 'block3_q_proj', 'transformer_h_3_attn_attention_k_proj': 'block3_k_proj', 'transformer_h_3_attn_attention_v_proj': 'block3_v_proj', 'matmul_6': 'block3_attn_qk_matmul', 'softmax_3': 'block3_attn_softmax', 'matmul_7': 'block3_attn_v_matmul', 'transformer_h_3_attn_attention_out_proj': 'block3_out_proj', 'add_37': 'block3_residual_attn', 'transformer_h_3_ln_2': 'block3_ln2', 'transformer_h_3_mlp_c_fc': 'block3_mlp_c_fc', 'block3_gelu': 'block3_gelu', 'transformer_h_3_mlp_c_proj': 'block3_mlp_c_proj', 'add_40': 'block3_residual_mlp', 'transformer_ln_f': 'final_ln', 'lm_head': 'lm_head'}

RESNET18_BLOCKS = [
    "layer1_0", "layer1_1",
    "layer2_0", "layer2_1",
    "layer3_0", "layer3_1",
    "layer4_0", "layer4_1",
]

RESNET50_BLOCKS = [
    *[f"layer1_{i}" for i in range(3)],
    *[f"layer2_{i}" for i in range(4)],
    *[f"layer3_{i}" for i in range(6)],
    *[f"layer4_{i}" for i in range(3)],
]

WANTED = {
    "model", "parameters", "dataset",
    "operator", "node_name",
    "operator_type", "operator_role", "block",
    "input_shape", "output_shape",
    "input_dtype", "output_dtype",
    "input_mb", "output_mb",
    "saved_tensor_count", "saved_tensor_shapes",
    "saved_tensor_dtypes", "saved_tensor_mb",
    "device", "batch_size", "sequence_length",
    "concurrency", "workers", "repeats_per_worker", "samples",
    "median_ms", "mean_ms", "p95_ms", "min_ms", "max_ms",
}


def paths_for_model(model):
    if model == "resnet18":
        return {
            "cpu": TEST_DIR / "test_operator_concurrency_resnet18_mp.py",
            "cuda": TEST_DIR / "test_op_con_nvtx_resnet18_mp.py",
            "list": TEST_DIR / "test_op_con_nvtx_resnet18.py",
        }
    if model == "resnet50":
        return {
            "cpu": TEST_DIR / "test_operator_concurrency_resnet50_mp.py",
            "cuda": TEST_DIR / "test_op_con_nvtx_resnet50_mp.py",
            "list": TEST_DIR / "test_op_con_nvtx_resnet50.py",
        }
    if model == "tinystories":
        nvtx_mp = TEST_DIR / "test_op_con_nvtx_tinystories_mp.py"
        regular_mp = TEST_DIR / "test_operator_concurrency_tinystories_mp.py"
        return {
            "cpu": regular_mp,
            "cuda": nvtx_mp if nvtx_mp.exists() else regular_mp,
            "list": None,
            "tinystories_nvtx_available": nvtx_mp.exists(),
        }
    raise ValueError(model)


def sanitize(name):
    return (
        name.replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def parse_stdout(stdout):
    parsed = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in WANTED:
            parsed[key] = value
    return parsed


def get_resnet_nodes(model, device, dataset, batch_size):
    p = paths_for_model(model)
    cmd = [
        sys.executable,
        str(p["list"]),
        "--device", device,
        "--dataset", dataset,
        "--batch-size", str(batch_size),
        "--list",
    ]

    print("[NODE LIST]")
    print(" ".join(cmd), flush=True)

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
        for part in line.split():
            if part.startswith("name="):
                nodes.append(part.split("=", 1)[1])
                break

    if not nodes:
        raise RuntimeError("No FX nodes parsed.")

    return nodes


def semantic_map(model, nodes):
    if model == "tinystories":
        return {
            n: TINYSTORIES_SHORT_NAMES.get(n, n)
            for n in nodes
        }

    blocks = RESNET18_BLOCKS if model == "resnet18" else RESNET50_BLOCKS

    out = {}
    add_idx = 0
    for node in nodes:
        if node == "add" or node.startswith("add_"):
            out[node] = (
                f"{blocks[add_idx]}_residual_add"
                if add_idx < len(blocks)
                else node
            )
            add_idx += 1
        else:
            out[node] = node
    return out


def run_profile(
    *,
    model,
    node,
    device,
    dataset,
    batch_size,
    sequence_length,
    concurrency,
    repeats,
    warmup,
    nsys_bin,
    report_dir,
):
    p = paths_for_model(model)
    profile_script = p[device]

    profile_cmd = [
        sys.executable,
        str(profile_script),
        "--device", device,
        "--batch-size", str(batch_size),
        "--node", node,
        "--concurrency", str(concurrency),
        "--repeats", str(repeats),
        "--warmup", str(warmup),
    ]

    if model == "tinystories":
        profile_cmd += [
            "--sequence-length", str(sequence_length),
        ]
    else:
        profile_cmd += [
            "--dataset", dataset,
        ]

    use_nsys = device == "cuda"

    if model == "tinystories" and not p.get(
        "tinystories_nvtx_available",
        False,
    ):
        use_nsys = False
        print(
            "[WARNING] tests/test_op_con_nvtx_tinystories_mp.py "
            "not found; TinyStories CUDA falls back to regular MP timing."
        )

    report_path = ""

    if use_nsys:
        report_prefix = (
            report_dir
            / (
                f"{model}_{sanitize(node)}_"
                f"{device}_con{concurrency}"
            )
        )

        cmd = [
            nsys_bin,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--force-overwrite=true",
            "-o", str(report_prefix),
            *profile_cmd,
        ]

        report_path = str(report_prefix) + ".qdstrm"

    else:
        cmd = profile_cmd

    print()
    print("[RUN]")
    print(" ".join(cmd), flush=True)

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = parse_stdout(result.stdout)

    if "node_name" not in parsed or "median_ms" not in parsed:
        raise RuntimeError(
            f"Could not parse result for {node}.\n"
            f"stdout:\n{result.stdout}"
        )

    parsed["nsys_rep"] = report_path
    parsed["warmup"] = str(warmup)
    return parsed


def add_slowdown(rows):
    baseline = {}

    for row in rows:
        try:
            if int(row["concurrency"]) != 1:
                continue
            key = (
                row["device"],
                row["operator"],
                row.get("dataset", ""),
                row.get("batch_size", ""),
                row.get("sequence_length", ""),
            )
            baseline[key] = float(row["median_ms"])
        except Exception:
            pass

    for row in rows:
        key = (
            row["device"],
            row["operator"],
            row.get("dataset", ""),
            row.get("batch_size", ""),
            row.get("sequence_length", ""),
        )
        base = baseline.get(key)
        try:
            med = float(row["median_ms"])
            row["slowdown_vs_con1"] = med / base if base else ""
        except Exception:
            row["slowdown_vs_con1"] = ""


def write_csv(path, rows):
    preferred = [
        "timestamp",
        "model", "parameters", "dataset",
        "operator", "node_name",
        "operator_type", "operator_role", "block",
        "input_shape", "output_shape",
        "input_dtype", "output_dtype",
        "input_mb", "output_mb",
        "saved_tensor_count", "saved_tensor_shapes",
        "saved_tensor_dtypes", "saved_tensor_mb",
        "device", "batch_size", "sequence_length",
        "concurrency", "workers", "repeats_per_worker", "samples",
        "median_ms", "mean_ms", "p95_ms", "min_ms", "max_ms",
        "slowdown_vs_con1", "warmup", "nsys_rep",
    ]

    seen = set()
    fields = []
    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
            seen.add(key)

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "resnet50", "tinystories"],
    )

    parser.add_argument(
        "--dataset",
        choices=["cifar10", "imagenet"],
        default="cifar10",
        help="Used by ResNet models.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)

    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "cuda"],
        default=["cuda"],
    )

    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    parser.add_argument("--nodes", nargs="+", default=None)
    parser.add_argument("--node-label", default="")

    parser.add_argument(
        "--output-dir",
        default="./exp_csv/operator_concurrency_nvtx",
    )

    parser.add_argument("--nsys-bin", default="nsys")

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    human_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_root = Path(args.output_dir)
    reports_root = (
        Path("exp_nsys")
        / args.model
    )

    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    failures = []

    for device in args.devices:
        if args.nodes:
            nodes = list(args.nodes)
        elif args.model == "tinystories":
            nodes = list(TINYSTORIES_DEFAULT_NODES)
        else:
            nodes = get_resnet_nodes(
                args.model,
                device,
                args.dataset,
                args.batch_size,
            )

        semantic = semantic_map(args.model, nodes)
        total = len(nodes) * len(args.concurrencies)
        current = 0

        print("=" * 100)
        print("Unified NVTX Operator Concurrency Sweep")
        print("=" * 100)
        print(f"model           = {args.model}")
        print(f"device          = {device}")
        print(f"dataset         = {args.dataset if args.model != 'tinystories' else 'N/A'}")
        print(f"batch_size      = {args.batch_size}")
        print(f"sequence_length = {args.sequence_length if args.model == 'tinystories' else 'N/A'}")
        print(f"repeats         = {args.repeats}")
        print(f"warmup          = {args.warmup}")
        print(f"concurrencies   = {args.concurrencies}")
        print(f"nodes           = {len(nodes)}")
        print("=" * 100)

        for node in nodes:
            for concurrency in args.concurrencies:
                current += 1

                print(
                    f"[{current}/{total}] "
                    f"operator={semantic[node]} "
                    f"node={node} con={concurrency}"
                )

                try:
                    row = run_profile(
                        model=args.model,
                        node=node,
                        device=device,
                        dataset=args.dataset,
                        batch_size=args.batch_size,
                        sequence_length=args.sequence_length,
                        concurrency=concurrency,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        nsys_bin=args.nsys_bin,
                        report_dir=reports_root,
                    )

                    row["timestamp"] = human_timestamp
                    row["model"] = (
                        TINYSTORIES_MODEL_NAME
                        if args.model == "tinystories"
                        else args.model
                    )
                    if args.model == "tinystories":
                        row.setdefault(
                            "parameters",
                            str(TINYSTORIES_PARAMETERS),
                        )
                        row["dataset"] = ""
                        row["sequence_length"] = str(args.sequence_length)
                    else:
                        row["dataset"] = args.dataset
                        row["sequence_length"] = ""

                    row["operator"] = semantic[node]
                    row["batch_size"] = str(args.batch_size)
                    row["device"] = device

                    rows.append(row)

                    print(
                        f"[OK] median={row['median_ms']} ms "
                        f"report={row.get('nsys_rep', '')}"
                    )

                except Exception as e:
                    print(
                        f"[FAIL] node={node} con={concurrency} "
                        f"error={e}"
                    )
                    failures.append((device, node, concurrency, str(e)))

                    if not args.continue_on_error:
                        raise

    add_slowdown(rows)

    label_tag = f"_{args.node_label}" if args.node_label else ""
    shape_tag = (
        f"seq{args.sequence_length}"
        if args.model == "tinystories"
        else args.dataset
    )
    device_tag = "-".join(args.devices)

    output = (
        output_root
        / (
            f"{timestamp}{label_tag}_{args.model}_"
            f"{shape_tag}_bs{args.batch_size}_"
            f"{device_tag}_nvtx.csv"
        )
    )

    write_csv(output, rows)

    print()
    print("=" * 100)
    print("SWEEP COMPLETE")
    print("=" * 100)
    print(f"rows     = {len(rows)}")
    print(f"failures = {len(failures)}")
    print(f"csv      = {output}")
    print(f"reports  = {reports_root}")

    if failures:
        print("FAILED RUNS:")
        for item in failures:
            print(item)


if __name__ == "__main__":
    main()
