import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecomputeCost:
    key: str
    recompute_ms: float
    output_to_cpu_ms: float
    patch_ms: float 
    start: str
    target: str
    path: tuple[str, ...]

@dataclass(frozen=True)
class RecomputeCalibration:
    operator_scale: float
    recompute_fixed_ms: float
    inject_ms_per_mb: float

def load_recompute_calibration(
    json_path: str | Path,
) -> RecomputeCalibration:
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file does not exist: {path}"
        )

    with path.open("r") as f:
        raw = json.load(f)

    return RecomputeCalibration(
        operator_scale=float(
            raw.get("operator_scale", 1.0)
        ),
        recompute_fixed_ms=float(
            raw.get("recompute_fixed_ms", 0.0)
        ),
        inject_ms_per_mb=float(
            raw.get("inject_ms_per_mb", 0.0)
        ),
    )

def load_recompute_cost_table(
    csv_path: str | Path,
) -> dict[str, RecomputeCost]:
    """
    Expected CSV columns:
        key,recompute_ms,...
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Recompute cost profile does not exist: {path}"
        )

    costs: dict[str, RecomputeCost] = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "key",
            "recompute_ms",
            "output_to_cpu_ms",
            "patch_ms",
            "start",
            "target",
            "path",
        }
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Recompute cost profile is missing required "
                f"columns: {sorted(missing)}"
            )

        for row in reader:
            key = (row.get("key") or "").strip()

            if not key:
                continue

            recompute_ms_raw = (
                row.get("recompute_ms") or ""
            ).strip()

            output_to_cpu_ms_raw = (
                row.get("output_to_cpu_ms") or ""
            ).strip()

            patch_ms_raw = (
                row.get("patch_ms") or ""
            ).strip()

            inject_missing_raw = (
                row.get("inject_profile_missing") or "0"
            ).strip()

            output_missing = int(
                float(
                    row.get(
                        "output_profile_missing",
                        "0",
                    )
                )
            )

            try:
                inject_missing = int(
                    float(inject_missing_raw)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid inject_profile_missing="
                    f"{inject_missing_raw!r} "
                    f"for key={key!r}"
                ) from exc

            if inject_missing:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=inject_profile_missing",
                    flush=True,
                )
                continue

            if output_missing:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=output_profile_missing",
                    flush=True,
                )
                continue

            if not recompute_ms_raw:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=empty_recompute_ms",
                    flush=True,
                )
                continue

            if not output_to_cpu_ms_raw:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=empty_output_to_cpu_ms",
                    flush=True,
                )
                continue

            if not patch_ms_raw:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=empty_patch_ms",
                    flush=True,
                )
                continue

            try:
                recompute_ms = float(
                    recompute_ms_raw
                )
                output_to_cpu_ms = float(
                    output_to_cpu_ms_raw
                )
                patch_ms = float(
                    patch_ms_raw
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid cost for key={key!r}: "
                    f"recompute_ms={recompute_ms_raw!r}, "
                    f"output_to_cpu_ms={output_to_cpu_ms_raw!r}, "
                    f"patch_ms={patch_ms_raw!r}"
                ) from exc
            
            if (
                recompute_ms < 0
                or output_to_cpu_ms < 0
                or patch_ms < 0
            ):
                raise ValueError(
                    f"Costs must be non-negative "
                    f"for key={key!r}: "
                    f"recompute_ms={recompute_ms}, "
                    f"output_to_cpu_ms={output_to_cpu_ms}, "
                    f"patch_ms={patch_ms}"
                )

            start = (
                row.get("start") or ""
            ).strip()

            target = (
                row.get("target") or ""
            ).strip()

            path_text = (
                row.get("path") or ""
            ).strip()

            path_nodes = tuple(
                node.strip()
                for node in path_text.split("->")
                if node.strip()
            )

            if not start:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=empty_start",
                    flush=True,
                )
                continue

            if not target:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=empty_target",
                    flush=True,
                )
                continue

            if not path_nodes:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} "
                    f"reason=empty_path",
                    flush=True,
                )
                continue

            costs[key] = RecomputeCost(
                key=key,
                recompute_ms=recompute_ms,
                output_to_cpu_ms=output_to_cpu_ms,
                patch_ms=patch_ms,
                start=start,
                target=target,
                path=path_nodes,
            )

    if not costs:
        raise ValueError(
            f"Recompute cost profile is empty: {path}"
        )

    return costs


from collections.abc import Callable
from typing import Any

def auto_drop_by_cost(
    payload,
    candidate_keys,
    cost_table: dict[str, RecomputeCost],
    network_mbps: float,

    estimate_selected_cost: Callable[
        [set[str]],
        dict[str, Any],
    ],

    min_benefit_ms: float = 0.0,
    max_drop_ratio: float | None = None,
):
    """
    Drop candidate tensors when their estimated transmission time
    is greater than their measured recomputation time.

    This version evaluates candidates independently. Shared path and
    recomputation-cache effects are not yet included.
    """

    if network_mbps is None or network_mbps <= 0:
        raise ValueError(
            f"Network bandwidth must be positive: {network_mbps}"
        )

    payload.meta = getattr(payload, "meta", {})

    total_payload_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in payload.tensors.values()
    )

    max_drop_bytes = None

    if max_drop_ratio is not None:
        if not 0.0 <= max_drop_ratio <= 1.0:
            raise ValueError(
                f"max_drop_ratio must be in [0, 1], "
                f"got {max_drop_ratio}"
            )

        max_drop_bytes = int(
            total_payload_bytes * max_drop_ratio
        )

    rows = []
    missing_tensor_keys = []
    missing_profile_keys = []

    for key in candidate_keys:
        tensor = payload.tensors.get(key)

        if tensor is None:
            missing_tensor_keys.append(key)
            continue

        profile = cost_table.get(key)

        if profile is None:
            missing_profile_keys.append(key)
            continue

        nbytes = tensor.numel() * tensor.element_size()
        tensor_mb = nbytes / 1024 / 1024

        send_ms = (
            tensor_mb
            * 8.0
            * 1000.0
            / network_mbps
        )

        operator_ms = profile.recompute_ms

        predicted_output_to_cpu_ms = (
            profile.output_to_cpu_ms
        )

        predicted_patch_ms = profile.patch_ms

        # 이 값은 초기 정렬용 추정치다.
        initial_candidate_cost_ms = (
            operator_ms
            + predicted_output_to_cpu_ms
            + predicted_patch_ms
        )

        initial_benefit_ms = (
            send_ms
            - initial_candidate_cost_ms
        )

        rows.append({
            "key": key,
            "nbytes": nbytes,
            "tensor_mb": tensor_mb,
            "send_ms": send_ms,

            "operator_ms": operator_ms,
            "predicted_output_to_cpu_ms": (
                predicted_output_to_cpu_ms
            ),
            "predicted_patch_ms": predicted_patch_ms,
            "marginal_recompute_ms": (
                initial_candidate_cost_ms
            ),

            "benefit_ms": initial_benefit_ms,
        })

    rows.sort(
        key=lambda row: row["benefit_ms"],
        reverse=True,
    )

    ####

    selected_rows = []
    selected_keys: set[str] = set()
    selected_bytes = 0

    current_cost = {
        "operator_ms": 0.0,
        "output_to_cpu_ms": 0.0,
        "patch_ms": 0.0,
        "inject_ms": 0.0,
        "total_ms": 0.0,
        "missing": [],
    }

    current_send_saved_ms = 0.0
    current_benefit_ms = 0.0

    for row in rows:
        key = row["key"]

        would_exceed_limit = (
            max_drop_bytes is not None
            and selected_bytes + row["nbytes"]
            > max_drop_bytes
        )

        if would_exceed_limit:
            print(
                f"[AUTO_DROP_COST_REJECT] "
                f"key={key} "
                f"reason=max_drop_ratio",
                flush=True,
            )
            continue

        # 현재까지 선택한 후보 + 새 후보
        trial_keys = selected_keys | {key}

        # trial_keys 전체를 동시에 drop했을 때의
        # recompute + inject 비용
        trial_cost = estimate_selected_cost(
            trial_keys
        )

        missing = trial_cost.get("missing", [])

        if missing:
            print(
                f"[AUTO_DROP_COST_REJECT] "
                f"key={key} "
                f"reason=missing_cost "
                f"missing={missing[:10]}",
                flush=True,
            )
            continue

        trial_total_ms = float(
            trial_cost["total_ms"]
        )

        # 후보 하나를 추가함으로써 증가한 비용
        additional_total_cost_ms = (
            trial_total_ms
            - float(current_cost["total_ms"])
        )

        # 이 후보 tensor를 보내는 비용
        additional_send_ms = float(
            row["send_ms"]
        )

        # 핵심 판단
        #
        # 전송 비용이
        # 추가 recompute + inject 비용보다 크면
        # 해당 후보를 drop한다.
        if (
            additional_send_ms
            <= additional_total_cost_ms
        ):
            print(
                f"[AUTO_DROP_COST_REJECT] "
                f"key={key} "
                f"send_ms={additional_send_ms:.3f} "
                f"additional_cost_ms="
                f"{additional_total_cost_ms:.3f}",
                flush=True,
            )
            continue

        trial_send_saved_ms = (
            current_send_saved_ms
            + additional_send_ms
        )

        trial_benefit_ms = (
            trial_send_saved_ms
            - trial_total_ms
        )

        if trial_benefit_ms <= min_benefit_ms:
            print(
                f"[AUTO_DROP_COST_REJECT] "
                f"key={key} "
                f"reason=min_benefit "
                f"trial_benefit_ms="
                f"{trial_benefit_ms:.3f}",
                flush=True,
            )
            continue

        # 채택
        selected_keys = trial_keys
        selected_rows.append(row)
        selected_bytes += row["nbytes"]

        current_cost = trial_cost
        current_send_saved_ms = (
            trial_send_saved_ms
        )
        current_benefit_ms = (
            trial_benefit_ms
        )

        print(
            f"[AUTO_DROP_COST_ACCEPT] "
            f"key={key} "
            f"send_ms={additional_send_ms:.3f} "
            f"additional_cost_ms="
            f"{additional_total_cost_ms:.3f} "
            f"selected={len(selected_keys)} "
            f"total_cost_ms={trial_total_ms:.3f} "
            f"total_benefit_ms="
            f"{trial_benefit_ms:.3f}",
            flush=True,
        )


    predicted_send_saved_ms = float(
        current_send_saved_ms
    )

    predicted_operator_ms = float(
        current_cost["operator_ms"]
    )
    predicted_output_to_cpu_ms = float(
        current_cost.get(
            "output_to_cpu_ms",
            0.0,
        )
    )

    predicted_patch_ms = float(
        current_cost.get(
            "patch_ms",
            current_cost.get("inject_ms", 0.0),
        )
    )
    predicted_inject_ms = predicted_patch_ms

    predicted_recompute_ms = float(
        current_cost["total_ms"]
    )

    predicted_benefit_ms = float(
        current_benefit_ms
    )

    # 전체 선택 결과가 fixed cost까지 포함해서 유리하지 않으면
    # 아무것도 드롭하지 않는다.
    if predicted_benefit_ms <= min_benefit_ms:
        selected_rows = []
        selected_keys = set()
        selected_bytes = 0

        current_cost = {
            "operator_ms": 0.0,
            "output_to_cpu_ms": 0.0,
            "patch_ms": 0.0,
            "inject_ms": 0.0,
            "total_ms": 0.0,
            "missing": [],
        }

        current_send_saved_ms = 0.0
        current_benefit_ms = 0.0

        predicted_send_saved_ms = 0.0
        predicted_operator_ms = 0.0
        predicted_output_to_cpu_ms = 0.0
        predicted_patch_ms = 0.0
        predicted_inject_ms = 0.0
        predicted_recompute_ms = 0.0
        predicted_benefit_ms = 0.0

    dropped = []
    saved_bytes = 0

    for row in selected_rows:
        removed = payload.tensors.pop(
            row["key"],
            None,
        )

        if removed is None:
            raise RuntimeError(
                f"Selected tensor disappeared before drop: "
                f"{row['key']}"
            )

        dropped.append(row["key"])
        saved_bytes += row["nbytes"]

    decisions = [
        {
            **row,
            "decision": (
                "recompute"
                if row["key"] in selected_keys
                else "send"
            ),
        }
        for row in rows
    ]

    ####

    dropped_ratio = (
        saved_bytes / total_payload_bytes
        if total_payload_bytes > 0
        else 0.0
    )

    payload.meta["auto_dropped_keys"] = dropped
    payload.meta["selection_policy"] = "cost"
    payload.meta["network_mbps"] = float(network_mbps)
    payload.meta["drop_ratio"] = float(dropped_ratio)
    payload.meta["dropped_count"] = len(dropped)
    payload.meta["saved_mb"] = saved_bytes / 1024 / 1024
    payload.meta[
        "predicted_output_to_cpu_ms"
    ] = predicted_output_to_cpu_ms

    payload.meta[
        "predicted_patch_ms"
    ] = predicted_patch_ms

    payload.meta["predicted_send_saved_ms"] = (
        predicted_send_saved_ms
    )
    payload.meta["predicted_operator_ms"] = (
        predicted_operator_ms
    )

    payload.meta["predicted_inject_ms"] = (
        predicted_inject_ms
    )

    payload.meta["predicted_recompute_ms"] = (
        predicted_recompute_ms
    )
    payload.meta["predicted_benefit_ms"] = (
        predicted_benefit_ms
    )

    payload.meta["cost_decisions"] = decisions
    payload.meta["cost_missing_tensor_keys"] = (
        missing_tensor_keys
    )
    payload.meta["cost_missing_profile_keys"] = (
        missing_profile_keys
    )

    print(
        f"[AUTO_DROP_COST] "
        f"network_mbps={network_mbps:.3f} "
        f"candidates={len(rows)} "
        f"dropped={len(dropped)} "
        f"drop_ratio={dropped_ratio:.4f} "
        f"saved_mb={saved_bytes / 1024 / 1024:.3f} "
        f"predicted_send_saved_ms="
        f"{predicted_send_saved_ms:.3f} "
        f"predicted_recompute_ms="
        f"{predicted_recompute_ms:.3f} "
        f"predicted_benefit_ms="
        f"{predicted_benefit_ms:.3f} "
        f"missing_tensor={len(missing_tensor_keys)} "
        f"missing_profile={len(missing_profile_keys)}",
        flush=True,
    )

    return payload