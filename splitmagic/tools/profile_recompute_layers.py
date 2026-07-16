import argparse
import csv
from pathlib import Path

import torch
import time

from splitmagic.models import make_resnet18_cifar10
from splitmagic.fx_trace import build_fx_node_list_with_shapes
from splitmagic.recompute import FXRecomputeEngine


def make_model(name: str):
    if name == "resnet18":
        return make_resnet18_cifar10()

    raise ValueError(f"Unsupported model: {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")

    device = torch.device(args.device)

    model = make_model(args.model).to(device)
    model.train()

    sample = torch.randn(
        args.batch_size,
        3,
        32,
        32,
        device=device,
    )

    gm = torch.fx.symbolic_trace(model)

    node_values = {}

    with torch.no_grad():
        env = {}

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = sample

            elif node.op == "call_module":
                module = gm.get_submodule(str(node.target))
                args_value = torch.fx.node.map_arg(
                    node.args,
                    lambda n: env[n.name],
                )
                kwargs_value = torch.fx.node.map_arg(
                    node.kwargs,
                    lambda n: env[n.name],
                )
                env[node.name] = module(
                    *args_value,
                    **kwargs_value,
                )

            elif node.op == "call_function":
                args_value = torch.fx.node.map_arg(
                    node.args,
                    lambda n: env[n.name],
                )
                kwargs_value = torch.fx.node.map_arg(
                    node.kwargs,
                    lambda n: env[n.name],
                )
                env[node.name] = node.target(
                    *args_value,
                    **kwargs_value,
                )

            elif node.op == "call_method":
                args_value = torch.fx.node.map_arg(
                    node.args,
                    lambda n: env[n.name],
                )
                kwargs_value = torch.fx.node.map_arg(
                    node.kwargs,
                    lambda n: env[n.name],
                )
                self_value, *rest = args_value
                env[node.name] = getattr(
                    self_value,
                    node.target,
                )(*rest, **kwargs_value)

            elif node.op == "output":
                continue

            if node.name in env:
                node_values[node.name] = env[node.name]

    engine = FXRecomputeEngine(
        model=model,
        gm=gm,
        node_values=dict(node_values),
    )

    rows = []

    for node in gm.graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            continue

        input_nodes = [
            arg
            for arg in node.args
            if isinstance(arg, torch.fx.Node)
        ]

        if not input_nodes:
            continue

        input_name = input_nodes[0].name
        input_tensor = node_values.get(input_name)

        if not torch.is_tensor(input_tensor):
            continue

        try:
            if node.op == "call_module":
                module = gm.get_submodule(str(node.target))
                op_type = module.__class__.__name__

                with torch.no_grad():
                    for _ in range(args.warmup):
                        module(input_tensor)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    t0 = time.perf_counter()

                    for _ in range(args.repeat):
                        module(input_tensor)

                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    t1 = time.perf_counter()

                avg_ms = (
                    (t1 - t0)
                    * 1000.0
                    / args.repeat
                )
            else:
                path = [input_name, node.name]
                engine.profile_path(
                    start_tensor=input_tensor,
                    path=path,
                    repeat=args.repeat,
                    warmup=args.warmup,
                )
                avg_ms = engine.profile_db[node.name]
                op_type = str(node.target)

        except Exception as exc:
            print(
                f"[PROFILE_SKIP] node={node.name} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        rows.append({
            "node_name": node.name,
            "fx_op": node.op,
            "op_type": op_type,
            "avg_ms": avg_ms,
        })

        print(
            f"[PROFILE_OK] node={node.name} "
            f"avg_ms={avg_ms:.6f}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_name",
                "fx_op",
                "op_type",
                "avg_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[PROFILE_DONE] path={output} "
        f"rows={len(rows)}"
    )


if __name__ == "__main__":
    main()