#!/usr/bin/env python3

import argparse
import subprocess
import sys
import time


def run(cmd,env=None):
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def run_split_test(
    model_name,
    single_script,
    node_a_script,
    node_b_script,
    compare_script,
    server_wait=3,
):
    print("\n" + "=" * 60)
    print(f"[MODEL] {model_name}")
    print("=" * 60)

    print("\n[1/4] Single Baseline")
    run([sys.executable, single_script])

    print("\n[2/4] Start Node B")

    node_b = subprocess.Popen(
        [sys.executable, node_b_script],
    )

    try:
        time.sleep(server_wait)

        print("\n[3/4] Run Node A")
        run([sys.executable, node_a_script])

        print("\n[4/4] Compare")
        run([sys.executable, compare_script])

        print(f"\n[SUCCESS] {model_name}")

    finally:
        print(f"\n[STOP] Node B ({model_name})")

        node_b.terminate()

        try:
            node_b.wait(timeout=5)
        except subprocess.TimeoutExpired:
            node_b.kill()
            node_b.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet18", "vgg", "mobilenetv2","all"],
        default="all",
    )

    args = parser.parse_args()

    tests = {
        "resnet18": {
            "single": "tests/test_single_resnet18.py",
            "node_a": "tests/test_node_a_resnet18.py",
            "node_b": "tests/test_node_b_resnet18.py",
            "compare": "tests/test_compare_resnet18.py",
        },
        "vgg": {
            "single": "tests/test_single_vgg.py",
            "node_a": "tests/test_node_a_vgg.py",
            "node_b": "tests/test_node_b_vgg.py",
            "compare": "tests/test_compare_vgg.py",
        },
        "mobilenetv2": {
            "single": "tests/test_single_mobilenetv2.py",
            "node_a": "tests/test_node_a_mobilenetv2.py",
            "node_b": "tests/test_node_b_mobilenetv2.py",
            "compare": "tests/test_compare_mobilenetv2.py",
        },

    }

    if args.model == "all":
        models = ["resnet18", "vgg","mobilenetv2"]
    else:
        models = [args.model]

    failed = []

    for model in models:
        try:
            run_split_test(
                model_name=model,
                single_script=tests[model]["single"],
                node_a_script=tests[model]["node_a"],
                node_b_script=tests[model]["node_b"],
                compare_script=tests[model]["compare"],
            )
        except Exception as e:
            print(f"\n[FAILED] {model}")
            print(e)
            failed.append(model)

    print("\n" + "=" * 60)

    if not failed:
        print("[ALL PASSED]")
    else:
        print("[FAILED MODELS]")
        for model in failed:
            print(" -", model)

    print("=" * 60)


if __name__ == "__main__":
    main()