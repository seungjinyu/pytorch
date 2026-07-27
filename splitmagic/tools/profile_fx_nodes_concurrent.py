from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
import gc

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.fx as fx
import torch.nn as nn


# ============================================================
# 기본 설정
# ============================================================

DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 10, 16, 32]
DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 100

CSV_FIELDS = [
    "node_name",
    "avg_ms",
    "node_op",
    "node_target",
    "concurrency",

    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",

    "avg_makespan_ms",
    "median_makespan_ms",
    "p95_makespan_ms",

    "avg_launch_ms",
    "avg_wall_ms",

    "throughput_per_sec",

    "input_shapes",
    "output_shape",
    "dtype",
    "device",
]

# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class CapturedNode:
    node_name: str
    node_op: str
    node_target: str

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    output_shape: str


@dataclass
class RoundResult:
    request_latencies_ms: list[float]

    # GPU에서 concurrency개의 작업이 모두 끝날 때까지 걸린 시간
    makespan_ms: float

    # Python이 concurrency개의 연산을 stream에 enqueue한 시간
    launch_ms: float

    # enqueue 시작부터 GPU synchronize까지의 전체 host wall time
    wall_ms: float

# ============================================================
# 유틸리티
# ============================================================

def percentile(
    values: list[float],
    p: float,
) -> float:
    if not values:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (p / 100.0) * (
        len(sorted_values) - 1
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return sorted_values[lower]

    ratio = position - lower

    return (
        sorted_values[lower] * (1.0 - ratio)
        + sorted_values[upper] * ratio
    )


def tensor_shape_string(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))

    if isinstance(value, tuple):
        return "[" + ", ".join(
            tensor_shape_string(v)
            for v in value
        ) + "]"

    if isinstance(value, list):
        return "[" + ", ".join(
            tensor_shape_string(v)
            for v in value
        ) + "]"

    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key}: {tensor_shape_string(child)}"
            for key, child in value.items()
        ) + "}"

    return type(value).__name__


def detach_runtime_value(
    value: Any,
) -> Any:
    if torch.is_tensor(value):
        return value.detach()

    if isinstance(value, tuple):
        return tuple(
            detach_runtime_value(child)
            for child in value
        )

    if isinstance(value, list):
        return [
            detach_runtime_value(child)
            for child in value
        ]

    if isinstance(value, dict):
        return {
            key: detach_runtime_value(child)
            for key, child in value.items()
        }

    return value

def move_runtime_value(
    value: Any,
    device: torch.device,
) -> Any:
    if torch.is_tensor(value):
        value = value.detach()

        if value.device == device:
            return value

        return value.to(
            device=device,
            non_blocking=True,
        )

    if isinstance(value, tuple):
        return tuple(
            move_runtime_value(child, device)
            for child in value
        )

    if isinstance(value, list):
        return [
            move_runtime_value(child, device)
            for child in value
        ]

    if isinstance(value, dict):
        return {
            key: move_runtime_value(child, device)
            for key, child in value.items()
        }

    return value

# ============================================================
# 실제 forward 입력 캡처
# ============================================================

class NodeInputCaptureInterpreter(
    fx.Interpreter
):
    """
    실제 GraphModule forward를 한 번 실행하면서
    각 FX node에 전달된 concrete args/kwargs를 저장한다.
    """

    def __init__(
        self,
        module: fx.GraphModule,
    ):
        super().__init__(module)

        self.captured_nodes: dict[
            str,
            CapturedNode,
        ] = {}

    def run_node(
        self,
        node: fx.Node,
    ) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(
            node
        )

        output = super().run_node(node)

        if node.op in {
            "call_module",
            "call_function",
            "call_method",
        }:
            self.captured_nodes[node.name] = CapturedNode(
                node_name=node.name,
                node_op=node.op,
                node_target=str(node.target),
                args=detach_runtime_value(args),
                kwargs=detach_runtime_value(kwargs),
                output_shape=tensor_shape_string(
                    output
                ),
            )

        return output


def capture_fx_node_inputs(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
) -> tuple[
    fx.GraphModule,
    dict[str, CapturedNode],
]:
    """
    모델을 FX trace하고 실제 입력으로 한 번 실행하여
    node별 concrete 입력을 수집한다.
    """

    gm = fx.symbolic_trace(model)

    interpreter = NodeInputCaptureInterpreter(gm)

    with torch.inference_mode():
        interpreter.run(*example_inputs)

    return gm, interpreter.captured_nodes


# ============================================================
# FX node 실행
# ============================================================

class FXNodeExecutor:
    def __init__(
        self,
        gm: fx.GraphModule,
    ) -> None:
        self.gm = gm

        self.node_by_name: dict[str, fx.Node] = {
            node.name: node
            for node in gm.graph.nodes
        }

        self.module_by_target = {
            str(node.target): gm.get_submodule(
                str(node.target)
            )
            for node in gm.graph.nodes
            if node.op == "call_module"
        }

    def execute(
        self,
        captured: CapturedNode,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if captured.node_op == "call_module":
            module = self.module_by_target[
                captured.node_target
            ]

            return module(
                *args,
                **kwargs,
            )

        if captured.node_op == "call_function":
            node = self.node_by_name[
                captured.node_name
            ]

            return node.target(
                *args,
                **kwargs,
            )

        if captured.node_op == "call_method":
            if not args:
                raise RuntimeError(
                    "call_method node has no receiver: "
                    f"{captured.node_name}"
                )

            receiver = args[0]

            method = getattr(
                receiver,
                captured.node_target,
            )

            return method(
                *args[1:],
                **kwargs,
            )

        raise ValueError(
            f"Unsupported FX operation: "
            f"{captured.node_op}"
        )


# ============================================================
# 실행 가능한 node 필터
# ============================================================

def is_profileable_node(
    gm: fx.GraphModule,
    captured: CapturedNode,
) -> bool:
    """
    비용 DB에 포함할 node를 결정한다.

    get_attr, placeholder, output은 capture 단계에서 이미 제외된다.
    """

    if captured.node_op == "call_module":
        module = gm.get_submodule(
            captured.node_target
        )

        # Dropout은 train/eval 상태에 따라 의미가 달라지고,
        # 일반적인 recompute saved-tensor 경로에는 잘 등장하지 않는다.
        if isinstance(
            module,
            nn.Dropout,
        ):
            return False

        return True

    if captured.node_op == "call_function":
        return True

    if captured.node_op == "call_method":
        return True

    return False

def run_cpu_round(
    executor: FXNodeExecutor,
    captured: CapturedNode,
    shared_args: tuple[Any, ...],
    shared_kwargs: dict[str, Any],
    concurrency: int,
) -> RoundResult:
    request_latencies_ms: list[float] = []

    wall_start = time.perf_counter()
    outputs: list[Any] = []

    for _ in range(concurrency):
        request_start = time.perf_counter()

        output = executor.execute(
            captured=captured,
            args=shared_args,
            kwargs=shared_kwargs,
        )

        request_end = time.perf_counter()

        outputs.append(output)

        request_latencies_ms.append(
            (request_end - request_start)
            * 1000.0
        )

    wall_end = time.perf_counter()

    wall_ms = (
        wall_end - wall_start
    ) * 1000.0

    # global_end_event.synchronize()

    # outputs.clear()
    # del outputs

    # torch.cuda.synchronize(device)

    outputs.clear()

    return RoundResult(
        request_latencies_ms=request_latencies_ms,
        makespan_ms=wall_ms,
        launch_ms=wall_ms,
        wall_ms=wall_ms,
    )

def run_cuda_concurrent_round(
    executor: FXNodeExecutor,
    captured: CapturedNode,
    shared_args: tuple[Any, ...],
    shared_kwargs: dict[str, Any],
    device: torch.device,
    streams: list[torch.cuda.Stream],
) -> RoundResult:
    concurrency = len(streams)

    if concurrency <= 0:
        raise ValueError("streams must not be empty")

    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    request_start_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(concurrency)
    ]

    request_end_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(concurrency)
    ]

    global_start_event = torch.cuda.Event(
        enable_timing=True
    )

    global_end_event = torch.cuda.Event(
        enable_timing=True
    )

    control_stream = torch.cuda.current_stream(
        device=device
    )

    outputs: list[Any] = []

    wall_start = time.perf_counter()
    launch_start = time.perf_counter()

    global_start_event.record(control_stream)

    for index, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            stream.wait_event(global_start_event)

            request_start_events[index].record(stream)

            output = executor.execute(
                captured=captured,
                args=shared_args,
                kwargs=shared_kwargs,
            )

            outputs.append(output)

            request_end_events[index].record(stream)

    for end_event in request_end_events:
        control_stream.wait_event(end_event)

    global_end_event.record(control_stream)

    launch_end = time.perf_counter()

    global_end_event.synchronize()

    wall_end = time.perf_counter()

    request_latencies_ms = [
        float(start.elapsed_time(end))
        for start, end in zip(
            request_start_events,
            request_end_events,
        )
    ]

    makespan_ms = float(
        global_start_event.elapsed_time(
            global_end_event
        )
    )

    launch_ms = (
        launch_end - launch_start
    ) * 1000.0

    wall_ms = (
        wall_end - wall_start
    ) * 1000.0

    outputs.clear()

    return RoundResult(
        request_latencies_ms=request_latencies_ms,
        makespan_ms=makespan_ms,
        launch_ms=launch_ms,
        wall_ms=wall_ms,
    )

def run_concurrent_round(
    executor: FXNodeExecutor,
    captured: CapturedNode,
    shared_args: tuple[Any, ...],
    shared_kwargs: dict[str, Any],
    device: torch.device,
    concurrency: int,
    streams: list[torch.cuda.Stream] | None,
) -> RoundResult:
    if concurrency <= 0:
        raise ValueError(
            f"Invalid concurrency: {concurrency}"
        )

    if device.type == "cuda":
        if streams is None:
            raise RuntimeError(
                "CUDA streams are required"
            )

        return run_cuda_concurrent_round(
            executor=executor,
            captured=captured,
            shared_args=shared_args,
            shared_kwargs=shared_kwargs,
            device=device,
            streams=streams,
        )

    return run_cpu_round(
        executor=executor,
        captured=captured,
        shared_args=shared_args,
        shared_kwargs=shared_kwargs,
        concurrency=concurrency,
    )

def runtime_dtype_string(
    value: Any,
) -> str:
    if torch.is_tensor(value):
        return str(value.dtype)

    if isinstance(value, (tuple, list)):
        dtypes = {
            runtime_dtype_string(child)
            for child in value
            if torch.is_tensor(child)
            or isinstance(
                child,
                (tuple, list, dict),
            )
        }

        dtypes.discard("")

        return "|".join(
            sorted(dtypes)
        )

    if isinstance(value, dict):
        dtypes = {
            runtime_dtype_string(child)
            for child in value.values()
            if torch.is_tensor(child)
            or isinstance(
                child,
                (tuple, list, dict),
            )
        }

        dtypes.discard("")

        return "|".join(
            sorted(dtypes)
        )

    return ""

# ============================================================
# node 하나의 concurrency profile
# ============================================================

def profile_node_concurrency(
    executor: FXNodeExecutor,
    captured: CapturedNode,
    device: torch.device,
    concurrency: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    print(
        f"\n[NODE_PROFILE_START] "
        f"node={captured.node_name} "
        f"op={captured.node_op} "
        f"target={captured.node_target} "
        f"concurrency={concurrency}",
        flush=True,
    )

    # captured input은 모든 stream에서 공유한다.
    shared_args = move_runtime_value(
        captured.args,
        device,
    )

    shared_kwargs = move_runtime_value(
        captured.kwargs,
        device,
    )

    if device.type == "cuda":
        streams: list[torch.cuda.Stream] | None = [
            torch.cuda.Stream(
                device=device
            )
            for _ in range(concurrency)
        ]
    else:
        streams = None

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------
    with torch.inference_mode():
        for _ in range(warmup):
            run_concurrent_round(
                executor=executor,
                captured=captured,
                shared_args=shared_args,
                shared_kwargs=shared_kwargs,
                device=device,
                concurrency=concurrency,
                streams=streams,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------
    all_request_latencies_ms: list[float] = []
    makespans_ms: list[float] = []
    launch_times_ms: list[float] = []
    wall_times_ms: list[float] = []

    with torch.inference_mode():
        for repeat_index in range(repeat):
            result = run_concurrent_round(
                executor=executor,
                captured=captured,
                shared_args=shared_args,
                shared_kwargs=shared_kwargs,
                device=device,
                concurrency=concurrency,
                streams=streams,
            )

            all_request_latencies_ms.extend(
                result.request_latencies_ms
            )

            makespans_ms.append(
                result.makespan_ms
            )

            launch_times_ms.append(
                result.launch_ms
            )

            wall_times_ms.append(
                result.wall_ms
            )

            if (
                repeat_index == 0
                or (repeat_index + 1) % 10 == 0
                or repeat_index + 1 == repeat
            ):
                print(
                    f"[NODE_PROFILE_PROGRESS] "
                    f"node={captured.node_name} "
                    f"concurrency={concurrency} "
                    f"round={repeat_index + 1}/{repeat}",
                    flush=True,
                )

    avg_ms = statistics.mean(
        all_request_latencies_ms
    )

    avg_makespan_ms = statistics.mean(
        makespans_ms
    )

    avg_launch_ms = statistics.mean(
        launch_times_ms
    )

    avg_wall_ms = statistics.mean(
        wall_times_ms
    )

    if avg_makespan_ms <= 0:
        throughput_per_sec = 0.0
    else:
        throughput_per_sec = (
            concurrency * 1000.0
            / avg_makespan_ms
        )

    row = {
        "node_name": captured.node_name,
        "avg_ms": avg_ms,
        "node_op": captured.node_op,
        "node_target": captured.node_target,
        "concurrency": concurrency,

        "median_ms": statistics.median(
            all_request_latencies_ms
        ),
        "p95_ms": percentile(
            all_request_latencies_ms,
            95.0,
        ),
        "min_ms": min(
            all_request_latencies_ms
        ),
        "max_ms": max(
            all_request_latencies_ms
        ),

        "avg_makespan_ms": (
            avg_makespan_ms
        ),
        "median_makespan_ms": (
            statistics.median(
                makespans_ms
            )
        ),
        "p95_makespan_ms": percentile(
            makespans_ms,
            95.0,
        ),

        "avg_launch_ms": avg_launch_ms,
        "avg_wall_ms": avg_wall_ms,

        "throughput_per_sec": (
            throughput_per_sec
        ),

        "input_shapes": tensor_shape_string(
            shared_args
        ),
        "output_shape": (
            captured.output_shape
        ),
        "dtype": runtime_dtype_string(
            shared_args
        ),
        "device": str(device),
    }

    print(
        f"[NODE_PROFILE_RESULT] "
        f"node={captured.node_name} "
        f"concurrency={concurrency} "
        f"avg_ms={avg_ms:.6f} "
        f"makespan_ms={avg_makespan_ms:.6f} "
        f"launch_ms={avg_launch_ms:.6f} "
        f"wall_ms={avg_wall_ms:.6f} "
        f"throughput={throughput_per_sec:.2f}",
        flush=True,
    )

    del shared_args
    del shared_kwargs
    del streams

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    gc.collect()

    return row

# ============================================================
# CSV
# ============================================================

def save_rows(
    rows: list[dict[str, Any]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[PROFILE_CSV_SAVE] "
        f"path={csv_path} "
        f"rows={len(rows)}",
        flush=True,
    )


# ============================================================
# 전체 profile 함수
# ============================================================

def profile_fx_nodes_concurrent(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    csv_path: str | Path,
    concurrencies: Iterable[int] = (
        1,
        2,
        4,
        8,
        10,
        16,
        32,
        64,
        128,
        256
    ),
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    include_node_names: set[str] | None = None,
    exclude_node_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    외부에서 호출하는 main profiling 함수.

    model:
        실제 SplitMagic 모델

    example_inputs:
        실제 deployment와 같은 batch/shape 입력

    include_node_names:
        지정하면 해당 node만 측정

    exclude_node_names:
        지정된 node는 제외
    """

    if not example_inputs:
        raise ValueError(
            "example_inputs must not be empty"
        )

    first_tensor = next(
        (
            value
            for value in example_inputs
            if torch.is_tensor(value)
        ),
        None,
    )

    if first_tensor is None:
        raise ValueError(
            "example_inputs must contain a tensor"
        )

    device = first_tensor.device

    # 중요한 점:
    # 프로파일링 중 BN running statistics가 여러 thread에서
    # 동시에 변경되지 않도록 eval 모드로 측정한다.
    #
    # 현재 FX recompute가 train-mode BN을 그대로 실행한다면
    # 아래 model.eval() 정책은 별도로 맞춰야 한다.
    original_training = model.training
    model.eval()

    try:
        gm, captured_nodes = (
            capture_fx_node_inputs(
                model=model,
                example_inputs=example_inputs,
            )
        )

        gm.to(device)
        gm.eval()

        executor = FXNodeExecutor(
            gm=gm
        )

        rows: list[dict[str, Any]] = []

        for node in gm.graph.nodes:
            captured = captured_nodes.get(
                node.name
            )

            if captured is None:
                continue

            if not is_profileable_node(
                gm,
                captured,
            ):
                continue

            if (
                include_node_names is not None
                and captured.node_name
                not in include_node_names
            ):
                continue

            if (
                exclude_node_names is not None
                and captured.node_name
                in exclude_node_names
            ):
                continue

            for concurrency in concurrencies:
                if concurrency <= 0:
                    raise ValueError(
                        f"Invalid concurrency: "
                        f"{concurrency}"
                    )

                row = profile_node_concurrency(
                    executor=executor,
                    captured=captured,
                    device=device,
                    concurrency=concurrency,
                    warmup=warmup,
                    repeat=repeat,
                )

                rows.append(row)

                # 중간 실패에도 현재 결과가 남도록
                # 매 node/concurrency마다 저장
                save_rows(
                    rows=rows,
                    csv_path=Path(csv_path),
                )

        return rows

    finally:
        model.train(original_training)


# ============================================================
# ResNet18 CIFAR 실행 예제
# ============================================================

def build_resnet18_cifar(
    device: torch.device,
) -> nn.Module:
    """
    네 프로젝트의 실제 model builder가 있으면
    이 함수 대신 그것을 import해서 사용하면 된다.
    """

    from torchvision.models import resnet18

    model = resnet18(
        weights=None
    )

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        model.fc.in_features,
        10,
    )

    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCIES,
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(
            "./recompute_node_concurrent_profile.csv"
        ),
    )

    parser.add_argument(
        "--nodes",
        type=str,
        nargs="*",
        default=None,
        help=(
            "측정할 FX node 이름. "
            "생략하면 모든 실행 node 측정."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu","cuda"],
        default="cuda",
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, "
                "but CUDA is not available"
            )

        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_resnet18_cifar(
        device=device
    )

    x = torch.randn(
        args.batch_size,
        3,
        32,
        32,
        device=device,
    )

    include_node_names = (
        set(args.nodes)
        if args.nodes
        else None
    )

    print(
        f"[PROFILE_CONFIG] "
        f"device={device} "
        f"batch_size={args.batch_size} "
        f"concurrency={args.concurrency} "
        f"warmup={args.warmup} "
        f"repeat={args.repeat}",
        flush=True,
    )

    profile_fx_nodes_concurrent(
        model=model,
        example_inputs=(x,),
        csv_path=args.csv_path,
        concurrencies=args.concurrency,
        warmup=args.warmup,
        repeat=args.repeat,
        include_node_names=(
            include_node_names
        ),
    )


if __name__ == "__main__":
    main()