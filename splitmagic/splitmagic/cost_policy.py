import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecomputeCost:
    key: str
    recompute_ms: float
    
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

        required = {"key", "recompute_ms"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Recompute cost profile is missing required "
                f"columns: {sorted(missing)}"
            )

        for row in reader:
            key = (row.get("key") or "").strip()
            recompute_ms_raw = (
                row.get("recompute_ms") or ""
            ).strip()

            if not key:
                continue

            # 측정값이 비어 있는 candidate는 cost table에서 제외
            if not recompute_ms_raw:
                print(
                    f"[COST_TABLE_SKIP] "
                    f"key={key} reason=empty_recompute_ms",
                    flush=True,
                )
                continue

            try:
                recompute_ms = float(recompute_ms_raw)
            except ValueError as e:
                raise ValueError(
                    f"Invalid recompute_ms={recompute_ms_raw!r} "
                    f"for key={key!r}"
                ) from e

            if recompute_ms < 0:
                raise ValueError(
                    f"Recompute cost must be non-negative: "
                    f"{recompute_ms} for key={key}"
                )

            costs[key] = RecomputeCost(
                key=key,
                recompute_ms=recompute_ms,
            )

    if not costs:
        raise ValueError(
            f"Recompute cost profile is empty: {path}"
        )

    return costs


def auto_drop_by_cost(
    payload,
    candidate_keys,
    cost_table: dict[str, RecomputeCost],
    network_mbps: float,
    inject_ms_per_mb: float =0.0,
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

        predicted_inject_ms = (
            tensor_mb * inject_ms_per_mb
        )

        marginal_recompute_ms = (
            operator_ms
            + predicted_inject_ms
        )


        marginal_benefit_ms = (
            send_ms
            - marginal_recompute_ms
        )

        rows.append({
            "key": key,
            "nbytes": nbytes,
            "tensor_mb": tensor_mb,
            "send_ms": send_ms,

            "operator_ms": operator_ms,
            "predicted_inject_ms": predicted_inject_ms,
            "marginal_recompute_ms": marginal_recompute_ms,

            "benefit_ms": marginal_benefit_ms,
        })

    rows.sort(
        key=lambda row: row["benefit_ms"],
        reverse=True,
    )

    ####

    selected_rows = []
    selected_bytes = 0

    # 우선 fixed cost를 제외한 marginal benefit 기준으로 후보를 선택한다.
    for row in rows:
        if row["benefit_ms"] <= 0.0:
            continue

        would_exceed_limit = (
            max_drop_bytes is not None
            and selected_bytes + row["nbytes"] > max_drop_bytes
        )

        if would_exceed_limit:
            continue

        selected_rows.append(row)
        selected_bytes += row["nbytes"]


    predicted_send_saved_ms = sum(
        row["send_ms"]
        for row in selected_rows
    )

    #
    predicted_operator_ms = sum(
        row["operator_ms"]
        for row in selected_rows
    )

    predicted_inject_ms = sum(
        row["predicted_inject_ms"]
        for row in selected_rows
    )

    predicted_recompute_ms = (
        predicted_operator_ms
        + predicted_inject_ms
    )
    #

    predicted_benefit_ms = (
        predicted_send_saved_ms
        - predicted_recompute_ms
    )

    # 전체 선택 결과가 fixed cost까지 포함해서 유리하지 않으면
    # 아무것도 드롭하지 않는다.
    if predicted_benefit_ms <= min_benefit_ms:
        selected_rows = []
        selected_bytes = 0

        predicted_send_saved_ms = 0.0
        predicted_operator_ms = 0.0
        predicted_inject_ms = 0.0
        predicted_recompute_ms = 0.0
        predicted_benefit_ms = 0.0


    selected_keys = {
        row["key"]
        for row in selected_rows
    }

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