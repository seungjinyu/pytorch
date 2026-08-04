#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import json

from splitmagic.models import make_resnet18_cifar10
from splitmagic.runtime import (
    SplitRuntime,
    get_required_keys_from_plan,
    is_recomputable_key,
    read_dryrun_plan,
)


FINAL_FIELDS = [
    "key",
    "tensor_mb",
    "start",
    "target",
    "path_len",
    "recompute_ms",
    "missing_profile",
    "path",
    "output_to_cpu_ms",
    "patch_ms",
    "candidate_cost_ms",
    "output_profile_missing",
    "inject_profile_missing",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "resnet18":
        return make_resnet18_cifar10()

    raise ValueError(
        f"Unsupported model={model_name}. "
        "현재 스크립트는 resnet18부터 지원한다."
    )


def load_inject_profile(
    path: Path,
) -> dict[str, float]:
    prices: dict[str, float] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {"key", "patch_ms"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"Inject profile columns missing: "
                f"{sorted(missing)}"
            )

        for row in reader:
            key = (row.get("key") or "").strip()
            value = (row.get("patch_ms") or "").strip()

            if not key or not value:
                continue

            prices[key] = float(value)

    return prices

def load_output_profile(
    path: Path,
) -> dict[str, float]:
    prices: dict[str, float] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "key",
            "output_to_cpu_ms",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                f"Output-to-CPU profile columns missing: "
                f"{sorted(missing)}"
            )

        for row in reader:
            key = (
                row.get("key") or ""
            ).strip()

            value = (
                row.get(
                    "output_to_cpu_ms"
                )
                or ""
            ).strip()

            if not key or not value:
                continue

            prices[key] = float(value)

    return prices

def attach_runtime_costs(
    recompute_menu_path: Path,
    inject_profile_path: Path,
    output_profile_path: Path,
    inject_menu_path: Path,
    output_menu_path: Path,
    final_menu_path: Path,
) -> None:
    inject_prices = load_inject_profile(
        inject_profile_path
    )

    output_prices = load_output_profile(
        output_profile_path
    )

    inject_menu_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with inject_menu_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["key", "patch_ms"],
        )
        writer.writeheader()

        for key in sorted(inject_prices):
            writer.writerow({
                "key": key,
                "patch_ms": f"{inject_prices[key]:.9f}",
            })
    output_menu_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_menu_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "key",
                "output_to_cpu_ms",
            ],
        )

        writer.writeheader()

        for key in sorted(output_prices):
            writer.writerow({
                "key": key,
                "output_to_cpu_ms": (
                    f"{output_prices[key]:.9f}"
                ),
            })

    with recompute_menu_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        recompute_rows = list(
            csv.DictReader(file)
        )

    final_rows = []

    for source in recompute_rows:
        key = (source.get("key") or "").strip()

        recompute_text = (
            source.get("recompute_ms") or ""
        ).strip()

        recompute_ms = (
            float(recompute_text)
            if recompute_text
            else None
        )
        output_to_cpu_ms = output_prices.get(
            key
        )

        patch_ms = inject_prices.get(key)

        candidate_cost_ms = (
            recompute_ms
            + output_to_cpu_ms
            + patch_ms
            if (
                recompute_ms is not None
                and output_to_cpu_ms is not None
                and patch_ms is not None
            )
            else None
        )

        final_rows.append({
            "key": key,
            "tensor_mb": source.get(
                "tensor_mb",
                "",
            ),
            "start": source.get("start", ""),
            "target": source.get("target", ""),
            "path_len": source.get(
                "path_len",
                "",
            ),
            "recompute_ms": recompute_text,
            "missing_profile": source.get(
                "missing_profile",
                "",
            ),
            "path": source.get("path", ""),
            "output_to_cpu_ms": (
                f"{output_to_cpu_ms:.9f}"
                if output_to_cpu_ms is not None
                else ""
            ),
            "patch_ms": (
                f"{patch_ms:.9f}"
                if patch_ms is not None
                else ""
            ),
            "candidate_cost_ms": (
                f"{candidate_cost_ms:.9f}"
                if candidate_cost_ms is not None
                else ""
            ),
            "output_profile_missing": (
                0
                if output_to_cpu_ms is not None
                else 1
            ),
            "inject_profile_missing": (
                0
                if patch_ms is not None
                else 1
            ),
        })

    final_menu_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with final_menu_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FINAL_FIELDS,
        )
        writer.writeheader()
        writer.writerows(final_rows)
    missing_output = sum(
        int(row["output_profile_missing"])
        for row in final_rows
    )

    missing_recompute = sum(
        bool(
            str(row["missing_profile"]).strip()
        )
        or not str(row["recompute_ms"]).strip()
        for row in final_rows
    )

    missing_inject = sum(
        int(row["inject_profile_missing"])
        for row in final_rows
    )

    print(
        "[FINAL_MENU] "
        f"rows={len(final_rows)} "
        f"missing_recompute={missing_recompute} "
        f"missing_output={missing_output} "
        f"missing_inject={missing_inject} "
        f"path={final_menu_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build recompute/inject/final menus locally "
            "without Node A/B processes, ZMQ, or templates."
        )
    )

    parser.add_argument(
        "--model",
        choices=["resnet18"],
        required=True,
    )

    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="jin_template_plan_a.tsv",
    )

    parser.add_argument(
        "--recompute-profile",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--inject-profile",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-profile",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
    "--network-mbps",
    type=float,
    default=None,
    )

    parser.add_argument(
        "--mps-percent",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--profile-steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--metric",
        choices=["avg", "median"],
        default="median",
    )

    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    args = parser.parse_args()

    if abs(args.drop_ratio - 0.99) > 1e-9:
        raise ValueError(
            "Offline final menu must be built from "
            "drop_ratio=0.99 profiling. "
            f"received={args.drop_ratio}"
        )

    plan_path = args.plan.expanduser().resolve()
    
    recompute_profile = (
        args.recompute_profile
        .expanduser()
        .resolve()
    )
    output_profile = (
        args.output_profile
        .expanduser()
        .resolve()
    )
    inject_profile = (
        args.inject_profile
        .expanduser()
        .resolve()
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    for path in (
        plan_path,
        recompute_profile,
        inject_profile,
        output_profile,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_all(args.seed)

    device = torch.device("cpu")

    model = build_model(args.model)
    model.to(device)
    model.train()

    plan = read_dryrun_plan(
        str(plan_path)
    )

    if not plan:
        raise RuntimeError(
            f"Dry-run plan is empty: {plan_path}"
        )

    print(
        "[OFFLINE_CONFIG] "
        f"model={args.model} "
        f"batch_size={args.batch_size} "
        f"plan_rows={len(plan)} "
        f"recompute_profile={recompute_profile} "
        f"inject_profile={inject_profile}"
        f"output_profile={output_profile} ",
        flush=True,
    )

    dummy_x = torch.randn(
        args.batch_size,
        3,
        32,
        32,
        device=device,
    )

    # Node A 프로세스를 실행하는 것이 아니다.
    # 기존 capture 함수만 로컬 함수처럼 재사용한다.
    capture_runtime = SplitRuntime(
        model=model,
        role="A",
    )

    payload = (
        capture_runtime.capture_jin_forward_plan(
            x=dummy_x,
            plan=plan,
        )
    )

    required_keys = get_required_keys_from_plan(
        plan
    )

    candidate_keys = sorted(
        key
        for key in required_keys
        if (
            is_recomputable_key(key)
            and key in payload.tensors
        )
    )

    print(
        "[OFFLINE_CANDIDATES] "
        f"required_keys={len(required_keys)} "
        f"payload_keys={len(payload.tensors)} "
        f"candidate_keys={len(candidate_keys)}",
        flush=True,
    )

    if not candidate_keys:
        raise RuntimeError(
            "No recompute candidates were created"
        )

    # 같은 프로세스 안에서 기존 menu builder 재사용.
    menu_runtime = SplitRuntime(
        model=model,
        role="B",
    )

    recompute_menu_path = (
        output_dir / "recompute_menu.csv"
    )

    inject_menu_path = (
        output_dir / "inject_menu.csv"
    )
    output_menu_path = (
        output_dir / "output_to_cpu_menu.csv"
    )

    final_menu_path = (
        output_dir / "final_menu.csv"
    )

    menu_runtime.build_recompute_menu(
        candidate_keys=candidate_keys,
        payload=payload,
        device=device,
        profile_csv=str(recompute_profile),
        menu_csv=str(recompute_menu_path),
    )

    attach_runtime_costs(
        recompute_menu_path=recompute_menu_path,
        inject_profile_path=inject_profile,
        output_profile_path=output_profile,
        inject_menu_path=inject_menu_path,
        output_menu_path=output_menu_path,
        final_menu_path=final_menu_path,
    )
    metadata_path = (
        output_dir / "metadata.json"
    )

    metadata = {
        "model": args.model,
        "network_mbps": args.network_mbps,
        "mps_percent": args.mps_percent,
        "profile_steps": args.profile_steps,
        "metric": args.metric,
        "profiling_drop_ratio": (
            args.drop_ratio
        ),
        "batch_size": args.batch_size,
        "plan_path": str(plan_path),
        "recompute_profile_path": str(
            recompute_profile
        ),
        "inject_profile_path": str(
            inject_profile
        ),
        "output_to_cpu_profile_path": str(
            output_profile
        ),
        "final_menu_path": str(
            final_menu_path
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    print(
        f"[METADATA] path={metadata_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise