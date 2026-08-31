#!/usr/bin/env python3

"""
Unified operator-workload profiler for:

  1) ResNet-18
  2) roneneldan/TinyStories-33M

Goals
-----
- Use semantic operator names that can be joined with the existing latency CSVs.
- Capture:
    input/output shapes and MB
    autograd saved tensors
    operation count + unit + method
- Keep different work units explicit:
    FLOPs
    comparisons
    additions
    elements
    lookups

Important
---------
- fvcore is used where it provides a supported FLOP count.
- ReLU / MaxPool / residual Add are counted with explicit custom units.
- TinyStories semantic operators are reduced to the same 60-operator view used
  by the latency profiler:
    2 embeddings
    14 operators x 4 transformer blocks
    final LayerNorm
    LM head
- GELU is treated as one semantic operator per block.
"""

from datetime import datetime
import argparse
import csv
import math
import operator
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.fx as fx

from torchvision.models import resnet18
from fvcore.nn import FlopCountAnalysis

from transformers import AutoModelForCausalLM
from transformers.utils.fx import symbolic_trace


TINYSTORIES_MODEL_NAME = "roneneldan/TinyStories-33M"


# ============================================================
# Generic helpers
# ============================================================

def tensor_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / (1024.0 * 1024.0)


def collect_tensors(x: Any) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []

    if isinstance(x, torch.Tensor):
        tensors.append(x)

    elif isinstance(x, (tuple, list)):
        for v in x:
            tensors.extend(collect_tensors(v))

    elif isinstance(x, dict):
        for v in x.values():
            tensors.extend(collect_tensors(v))

    return tensors


def format_shapes(tensors: List[torch.Tensor]) -> str:
    return "; ".join(str(tuple(t.shape)) for t in tensors)


def format_dtypes(tensors: List[torch.Tensor]) -> str:
    return "; ".join(str(t.dtype) for t in tensors)


def total_mb(tensors: List[torch.Tensor]) -> float:
    return sum(tensor_mb(t) for t in tensors)


def total_numel(tensors: List[torch.Tensor]) -> int:
    return sum(t.numel() for t in tensors)


def clone_for_semantic(x: torch.Tensor) -> torch.Tensor:
    """
    Make an independent tensor for semantic-op replay.

    Floating-point tensors require grad so saved_tensors_hooks sees
    the autograd state used by the operation.
    """
    y = x.detach().clone()

    if y.is_floating_point():
        y.requires_grad_(True)

    return y


def make_row(
    *,
    operator_name: str,
    node_name: str,
    fx_op: str,
    target: str,
    operator_type: str,
    operator_role: str,
    block: str,
    input_tensors: List[torch.Tensor],
    output_tensors: List[torch.Tensor],
    saved_tensors: List[torch.Tensor],
    operation_count: Optional[float],
    operation_unit: str,
    operation_method: str,
) -> Dict[str, Any]:

    return {
        "operator": operator_name,
        "node_name": node_name,
        "fx_op": fx_op,
        "target": target,
        "operator_type": operator_type,
        "operator_role": operator_role,
        "block": block,

        "input_shape": format_shapes(input_tensors),
        "output_shape": format_shapes(output_tensors),

        "input_dtype": format_dtypes(input_tensors),
        "output_dtype": format_dtypes(output_tensors),

        "input_mb": total_mb(input_tensors),
        "output_mb": total_mb(output_tensors),

        "saved_tensor_count": len(saved_tensors),
        "saved_tensor_shapes": format_shapes(saved_tensors),
        "saved_tensor_dtypes": format_dtypes(saved_tensors),
        "saved_tensor_mb": total_mb(saved_tensors),

        "operation_count": operation_count,
        "operation_count_m": (
            operation_count / 1e6
            if operation_count is not None
            else None
        ),
        "operation_unit": operation_unit,
        "operation_method": operation_method,
    }


# ============================================================
# Per-FX-node capture
# ============================================================

class FXNodeCaptureInspector(fx.Interpreter):
    """
    Executes an FX graph once and records concrete inputs, outputs,
    and tensors saved by autograd for each FX node.
    """

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)
        self.records: Dict[str, Dict[str, Any]] = {}
        self.execution_order: List[str] = []

    def run_node(self, node: fx.Node):
        if node.op not in {
            "call_module",
            "call_function",
            "call_method",
        }:
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)

        input_tensors = (
            collect_tensors(args)
            + collect_tensors(kwargs)
        )

        saved_tensors: List[torch.Tensor] = []

        def pack_hook(tensor):
            saved_tensors.append(tensor)
            return tensor

        def unpack_hook(tensor):
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(
            pack_hook,
            unpack_hook,
        ):
            result = super().run_node(node)

        output_tensors = collect_tensors(result)

        self.execution_order.append(node.name)

        self.records[node.name] = {
            "node": node,
            "args": args,
            "kwargs": kwargs,
            "input_tensors": input_tensors,
            "output_tensors": output_tensors,
            "saved_tensors": saved_tensors,
            "result": result,
        }

        return result


# ============================================================
# fvcore helpers
# ============================================================

def safe_fvcore_analysis(
    model: nn.Module,
    inputs,
):
    try:
        analysis = FlopCountAnalysis(model, inputs)

        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)

        by_module = analysis.by_module()
        by_operator = analysis.by_operator()
        unsupported = analysis.unsupported_ops()

        return by_module, by_operator, unsupported

    except Exception as e:
        print(f"[WARNING] fvcore analysis failed: {e}")
        return {}, {}, {}


def lookup_fvcore_module(
    by_module: Dict[str, float],
    module_name: str,
) -> Optional[float]:

    candidates = [
        module_name,
        f"model.{module_name}",
    ]

    for name in candidates:
        if name in by_module:
            return float(by_module[name])

    return None


# ============================================================
# ResNet-18 semantic naming
# ============================================================

RESNET_BLOCKS = [
    "layer1_0",
    "layer1_1",
    "layer2_0",
    "layer2_1",
    "layer3_0",
    "layer3_1",
    "layer4_0",
    "layer4_1",
]


def resnet_block_from_name(name: str) -> str:
    m = re.match(r"(layer\d+_\d+)", name)
    return m.group(1) if m else "-1"


def resnet_semantic_metadata(
    gm: fx.GraphModule,
    node: fx.Node,
    add_index: int,
):
    """
    Returns:
      semantic_name, operator_type, operator_role, block,
      operation_unit, operation_method, next_add_index
    """

    name = node.name

    if node.op == "call_module":
        module = gm.get_submodule(str(node.target))

        if name == "conv1":
            return (
                "conv1", "Conv", "Stem Conv", "-1",
                "FLOPs", "fvcore", add_index
            )

        if name == "bn1":
            return (
                "bn1", "BatchNorm", "Stem BatchNorm", "-1",
                "FLOPs", "fvcore", add_index
            )

        if name == "relu":
            return (
                "relu", "ReLU", "Stem ReLU", "-1",
                "comparisons",
                "1 comparison/output element",
                add_index,
            )

        if name == "maxpool":
            return (
                "maxpool", "MaxPool", "Stem MaxPool", "-1",
                "comparisons",
                "(kernel elements - 1)/output element",
                add_index,
            )

        if name == "avgpool":
            return (
                "avgpool", "AdaptiveAvgPool", "Global AvgPool", "-1",
                "FLOPs", "fvcore", add_index
            )

        if name == "fc":
            return (
                "fc", "Linear", "Classifier", "-1",
                "FLOPs", "fvcore", add_index
            )

        block = resnet_block_from_name(name)

        if isinstance(module, nn.Conv2d):
            role = "Downsample Conv" if "downsample" in name else (
                "Conv1" if name.endswith("conv1") else "Conv2"
            )
            return (
                name, "Conv", role, block,
                "FLOPs", "fvcore", add_index
            )

        if isinstance(module, nn.BatchNorm2d):
            role = "Downsample BatchNorm" if "downsample" in name else (
                "BatchNorm1" if name.endswith("bn1") else "BatchNorm2"
            )
            return (
                name, "BatchNorm", role, block,
                "FLOPs", "fvcore", add_index
            )

        if isinstance(module, nn.ReLU):
            role = "Output ReLU" if name.endswith("_1") else "ReLU1"
            return (
                name, "ReLU", role, block,
                "comparisons",
                "1 comparison/output element",
                add_index,
            )

        return (
            name,
            module.__class__.__name__,
            module.__class__.__name__,
            block,
            "N/A",
            "unsupported",
            add_index,
        )

    # residual add
    if (
        node.op == "call_function"
        and node.target in (operator.add, torch.add)
    ) or (
        node.op == "call_method"
        and str(node.target) in {"add", "add_"}
    ):
        if add_index < len(RESNET_BLOCKS):
            block = RESNET_BLOCKS[add_index]
        else:
            block = f"residual_{add_index}"

        semantic_name = f"{block}_residual_add"

        return (
            semantic_name,
            "Add",
            "Residual Add",
            block,
            "additions",
            "1 addition/output element",
            add_index + 1,
        )

    # flatten is retained for transparency but has no arithmetic count
    if node.op == "call_function" and "flatten" in str(node.target):
        return (
            "flatten",
            "Flatten",
            "Flatten",
            "-1",
            "N/A",
            "view/reshape",
            add_index,
        )

    return (
        name,
        str(node.target),
        str(node.target),
        resnet_block_from_name(name),
        "N/A",
        "unsupported",
        add_index,
    )


def resnet_custom_count(
    gm: fx.GraphModule,
    node: fx.Node,
    operator_type: str,
    output_tensors: List[torch.Tensor],
) -> Optional[float]:

    n_out = total_numel(output_tensors)

    if operator_type in {"ReLU", "Add"}:
        return float(n_out)

    if operator_type == "MaxPool":
        module = gm.get_submodule(str(node.target))
        kernel = module.kernel_size

        if isinstance(kernel, int):
            kh = kw = kernel
        else:
            kh, kw = kernel

        return float(
            n_out * (kh * kw - 1)
        )

    return None


def profile_resnet18(args):
    print("[MODEL] ResNet-18")

    model = resnet18(
        weights=None,
        num_classes=args.num_classes,
    )
    model.eval()

    x = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        requires_grad=True,
    )

    total_params = sum(p.numel() for p in model.parameters())

    fvcore_by_module, fvcore_by_operator, unsupported = (
        safe_fvcore_analysis(model, x)
    )

    gm = fx.symbolic_trace(model)
    gm.eval()

    inspector = FXNodeCaptureInspector(gm)
    output = inspector.run(x)
    _ = output

    rows: List[Dict[str, Any]] = []
    add_index = 0

    for node_name in inspector.execution_order:
        rec = inspector.records[node_name]
        node = rec["node"]

        (
            semantic_name,
            operator_type,
            operator_role,
            block,
            operation_unit,
            operation_method,
            add_index,
        ) = resnet_semantic_metadata(
            gm,
            node,
            add_index,
        )

        operation_count = None

        if node.op == "call_module" and operation_unit == "FLOPs":
            operation_count = lookup_fvcore_module(
                fvcore_by_module,
                str(node.target),
            )

        if operation_count is None:
            operation_count = resnet_custom_count(
                gm,
                node,
                operator_type,
                rec["output_tensors"],
            )

        row = make_row(
            operator_name=semantic_name,
            node_name=node.name,
            fx_op=node.op,
            target=str(node.target),
            operator_type=operator_type,
            operator_role=operator_role,
            block=block,
            input_tensors=rec["input_tensors"],
            output_tensors=rec["output_tensors"],
            saved_tensors=rec["saved_tensors"],
            operation_count=operation_count,
            operation_unit=operation_unit,
            operation_method=operation_method,
        )

        row.update({
            "model": "resnet18",
            "parameters": total_params,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "sequence_length": "",
        })

        rows.append(row)

    return rows, fvcore_by_operator, unsupported


# ============================================================
# TinyStories helpers
# ============================================================

def build_tinystories():
    model = AutoModelForCausalLM.from_pretrained(
        TINYSTORIES_MODEL_NAME
    )
    model.config.use_cache = False
    model.eval()
    return model


class TinyStoriesWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits


def tinystories_module_semantics(target: str):
    """
    Return semantic metadata for module-backed TinyStories operators.
    """

    # Embeddings
    if target.endswith("transformer.wte") or target == "transformer.wte":
        return (
            "embedding_token",
            "Embedding",
            "Token Embedding",
            "-1",
            "lookups",
            "1 lookup/input token",
        )

    if target.endswith("transformer.wpe") or target == "transformer.wpe":
        return (
            "embedding_position",
            "Embedding",
            "Position Embedding",
            "-1",
            "lookups",
            "1 lookup/input position",
        )

    # Final LN / LM head
    if target.endswith("transformer.ln_f") or target == "transformer.ln_f":
        return (
            "final_ln",
            "LayerNorm",
            "Final LayerNorm",
            "-1",
            "FLOPs",
            "fvcore",
        )

    if target.endswith("lm_head") or target == "lm_head":
        return (
            "lm_head",
            "Linear",
            "LM Head",
            "-1",
            "FLOPs",
            "fvcore",
        )

    # Transformer block modules
    m = re.search(r"transformer\.h\.(\d+)\.", target)
    if not m:
        return None

    b = int(m.group(1))
    prefix = f"block{b}"

    if target.endswith(".ln_1"):
        return (
            f"{prefix}_ln1",
            "LayerNorm",
            "Attention LayerNorm",
            str(b),
            "FLOPs",
            "fvcore",
        )

    if target.endswith(".attention.q_proj"):
        return (
            f"{prefix}_q_proj",
            "Linear",
            "Q Projection",
            str(b),
            "FLOPs",
            "fvcore",
        )

    if target.endswith(".attention.k_proj"):
        return (
            f"{prefix}_k_proj",
            "Linear",
            "K Projection",
            str(b),
            "FLOPs",
            "fvcore",
        )

    if target.endswith(".attention.v_proj"):
        return (
            f"{prefix}_v_proj",
            "Linear",
            "V Projection",
            str(b),
            "FLOPs",
            "fvcore",
        )

    if target.endswith(".attention.out_proj"):
        return (
            f"{prefix}_out_proj",
            "Linear",
            "Attention Output Projection",
            str(b),
            "FLOPs",
            "fvcore",
        )

    if target.endswith(".ln_2"):
        return (
            f"{prefix}_ln2",
            "LayerNorm",
            "MLP LayerNorm",
            str(b),
            "FLOPs",
            "fvcore",
        )

    if target.endswith(".mlp.c_fc"):
        return (
            f"{prefix}_mlp_c_fc",
            "Linear",
            "MLP FC",
            str(b),
            "FLOPs",
            "fvcore",
        )

    if target.endswith(".mlp.c_proj"):
        return (
            f"{prefix}_mlp_c_proj",
            "Linear",
            "MLP Projection",
            str(b),
            "FLOPs",
            "fvcore",
        )

    return None


def is_matmul_node(node: fx.Node) -> bool:
    text = str(node.target).lower()
    return (
        "matmul" in text
        or text.endswith(".bmm")
        or text.endswith(".mm")
        or str(node.target) in {"matmul", "bmm", "mm"}
    )


def is_softmax_node(node: fx.Node) -> bool:
    return "softmax" in str(node.target).lower()


def is_add_node(node: fx.Node) -> bool:
    if node.op == "call_function" and node.target in (
        operator.add,
        torch.add,
    ):
        return True

    if node.op == "call_method" and str(node.target) in {
        "add",
        "add_",
    }:
        return True

    return False


def matmul_fma_count(
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
) -> Optional[float]:
    """
    Count matrix multiply work using fvcore-compatible convention:
      1 fused multiply-add = 1 FLOP.
    """

    if len(inputs) < 2 or not outputs:
        return None

    a = inputs[0]
    b = inputs[1]
    out = outputs[0]

    if a.dim() < 2 or b.dim() < 2 or out.dim() < 2:
        return None

    m = int(a.shape[-2])
    k = int(a.shape[-1])
    n = int(b.shape[-1])

    batch = math.prod(out.shape[:-2]) if out.dim() > 2 else 1

    return float(batch * m * k * n)


def module_fallback_fma_count(
    module: nn.Module,
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
) -> Optional[float]:
    """
    Fallback for nn.Linear only if fvcore cannot provide a module value.
    Same convention as fvcore: one FMA = one FLOP.
    """

    if not isinstance(module, nn.Linear):
        return None

    if not outputs:
        return None

    out = outputs[0]

    output_elements = out.numel()
    n = module.out_features

    if n == 0:
        return None

    vectors = output_elements // n

    return float(
        vectors
        * module.in_features
        * module.out_features
    )


def capture_semantic_gelu(
    model,
    block: int,
    gelu_input: torch.Tensor,
):
    """
    Replay the entire HF NewGELUActivation as one semantic operator.
    """

    module = (
        model.transformer
        .h[block]
        .mlp
        .act
    )

    x = clone_for_semantic(gelu_input)

    saved_tensors: List[torch.Tensor] = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    def unpack_hook(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(
        pack_hook,
        unpack_hook,
    ):
        y = module(x)

    return (
        [x],
        collect_tensors(y),
        saved_tensors,
    )


def profile_tinystories(args):
    print("[MODEL] TinyStories-33M")

    model = build_tinystories()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    vocab_size = model.config.vocab_size

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            args.batch_size,
            args.sequence_length,
        ),
        dtype=torch.long,
    )

    attention_mask = torch.ones(
        (
            args.batch_size,
            args.sequence_length,
        ),
        dtype=torch.long,
    )

    wrapper = TinyStoriesWrapper(model)
    wrapper.eval()

    fvcore_by_module, fvcore_by_operator, unsupported = (
        safe_fvcore_analysis(
            wrapper,
            (
                input_ids,
                attention_mask,
            ),
        )
    )

    print("[FX] tracing TinyStories...")

    gm = symbolic_trace(
        model,
        input_names=[
            "input_ids",
            "attention_mask",
        ],
    )
    gm.eval()

    print("[FX] trace successful")

    inspector = FXNodeCaptureInspector(gm)
    output = inspector.run(
        input_ids,
        attention_mask,
    )
    _ = output

    # --------------------------------------------------------
    # Build module-backed semantic rows first.
    # Also remember graph positions for each block.
    # --------------------------------------------------------

    rows_by_operator: Dict[str, Dict[str, Any]] = {}
    order_index = {
        name: i
        for i, name in enumerate(inspector.execution_order)
    }

    block_anchor_positions: Dict[int, Dict[str, int]] = {
        b: {}
        for b in range(4)
    }

    c_fc_outputs: Dict[int, torch.Tensor] = {}

    for node_name in inspector.execution_order:
        rec = inspector.records[node_name]
        node = rec["node"]

        if node.op != "call_module":
            continue

        target = str(node.target)

        meta = tinystories_module_semantics(target)
        if meta is None:
            continue

        (
            semantic_name,
            operator_type,
            operator_role,
            block,
            operation_unit,
            operation_method,
        ) = meta

        operation_count = None

        if operation_unit == "FLOPs":
            operation_count = lookup_fvcore_module(
                fvcore_by_module,
                target,
            )

            if operation_count is None:
                module = gm.get_submodule(target)
                operation_count = module_fallback_fma_count(
                    module,
                    rec["input_tensors"],
                    rec["output_tensors"],
                )

                if operation_count is not None:
                    operation_method = (
                        "fvcore-compatible FMA=1 fallback"
                    )
                else:
                    operation_method = (
                        "fvcore unavailable/unsupported"
                    )

        elif operator_type == "Embedding":
            # Count lookup indices, not output embedding elements.
            operation_count = (
                float(rec["input_tensors"][0].numel())
                if rec["input_tensors"]
                else None
            )

        row = make_row(
            operator_name=semantic_name,
            node_name=node.name,
            fx_op=node.op,
            target=target,
            operator_type=operator_type,
            operator_role=operator_role,
            block=block,
            input_tensors=rec["input_tensors"],
            output_tensors=rec["output_tensors"],
            saved_tensors=rec["saved_tensors"],
            operation_count=operation_count,
            operation_unit=operation_unit,
            operation_method=operation_method,
        )

        rows_by_operator[semantic_name] = row

        if block != "-1":
            b = int(block)

            if semantic_name.endswith("_q_proj"):
                block_anchor_positions[b]["q"] = order_index[node_name]

            elif semantic_name.endswith("_v_proj"):
                block_anchor_positions[b]["v"] = order_index[node_name]

            elif semantic_name.endswith("_out_proj"):
                block_anchor_positions[b]["out"] = order_index[node_name]

            elif semantic_name.endswith("_ln2"):
                block_anchor_positions[b]["ln2"] = order_index[node_name]

            elif semantic_name.endswith("_mlp_c_fc"):
                block_anchor_positions[b]["c_fc"] = order_index[node_name]

                if rec["output_tensors"]:
                    c_fc_outputs[b] = rec["output_tensors"][0]

            elif semantic_name.endswith("_mlp_c_proj"):
                block_anchor_positions[b]["c_proj"] = order_index[node_name]

    # --------------------------------------------------------
    # Identify non-module semantic ops inside each block by
    # graph order between module anchors.
    # --------------------------------------------------------

    ordered_names = inspector.execution_order

    for b in range(4):
        anchors = block_anchor_positions[b]

        required = {"q", "v", "out", "ln2", "c_fc", "c_proj"}

        if not required.issubset(anchors):
            missing = sorted(required - set(anchors))
            print(
                f"[WARNING] block {b}: missing FX anchors {missing}"
            )
            continue

        # Attention primitive region:
        # after V projection and before output projection.
        attn_region = ordered_names[
            anchors["v"] + 1:
            anchors["out"]
        ]

        matmuls = [
            name
            for name in attn_region
            if is_matmul_node(
                inspector.records[name]["node"]
            )
        ]

        softmaxes = [
            name
            for name in attn_region
            if is_softmax_node(
                inspector.records[name]["node"]
            )
        ]

        if len(matmuls) >= 1:
            name = matmuls[0]
            rec = inspector.records[name]
            node = rec["node"]

            rows_by_operator[
                f"block{b}_attn_qk_matmul"
            ] = make_row(
                operator_name=f"block{b}_attn_qk_matmul",
                node_name=node.name,
                fx_op=node.op,
                target=str(node.target),
                operator_type="MatMul",
                operator_role="QK MatMul",
                block=str(b),
                input_tensors=rec["input_tensors"],
                output_tensors=rec["output_tensors"],
                saved_tensors=rec["saved_tensors"],
                operation_count=matmul_fma_count(
                    rec["input_tensors"],
                    rec["output_tensors"],
                ),
                operation_unit="FLOPs",
                operation_method="FMA=1 matrix-multiply count",
            )

        if len(softmaxes) >= 1:
            name = softmaxes[0]
            rec = inspector.records[name]
            node = rec["node"]

            rows_by_operator[
                f"block{b}_attn_softmax"
            ] = make_row(
                operator_name=f"block{b}_attn_softmax",
                node_name=node.name,
                fx_op=node.op,
                target=str(node.target),
                operator_type="Softmax",
                operator_role="Attention Softmax",
                block=str(b),
                input_tensors=rec["input_tensors"],
                output_tensors=rec["output_tensors"],
                saved_tensors=rec["saved_tensors"],
                operation_count=float(
                    total_numel(rec["output_tensors"])
                ),
                operation_unit="elements",
                operation_method="processed output elements",
            )

        if len(matmuls) >= 2:
            name = matmuls[1]
            rec = inspector.records[name]
            node = rec["node"]

            rows_by_operator[
                f"block{b}_attn_v_matmul"
            ] = make_row(
                operator_name=f"block{b}_attn_v_matmul",
                node_name=node.name,
                fx_op=node.op,
                target=str(node.target),
                operator_type="MatMul",
                operator_role="Attention-V MatMul",
                block=str(b),
                input_tensors=rec["input_tensors"],
                output_tensors=rec["output_tensors"],
                saved_tensors=rec["saved_tensors"],
                operation_count=matmul_fma_count(
                    rec["input_tensors"],
                    rec["output_tensors"],
                ),
                operation_unit="FLOPs",
                operation_method="FMA=1 matrix-multiply count",
            )

        # Attention residual Add:
        # after out_proj, before ln2
        attn_residual_region = ordered_names[
            anchors["out"] + 1:
            anchors["ln2"]
        ]

        attn_adds = [
            name
            for name in attn_residual_region
            if is_add_node(
                inspector.records[name]["node"]
            )
        ]

        if attn_adds:
            name = attn_adds[-1]
            rec = inspector.records[name]
            node = rec["node"]

            rows_by_operator[
                f"block{b}_residual_attn"
            ] = make_row(
                operator_name=f"block{b}_residual_attn",
                node_name=node.name,
                fx_op=node.op,
                target=str(node.target),
                operator_type="Add",
                operator_role="Attention Residual",
                block=str(b),
                input_tensors=rec["input_tensors"],
                output_tensors=rec["output_tensors"],
                saved_tensors=rec["saved_tensors"],
                operation_count=float(
                    total_numel(rec["output_tensors"])
                ),
                operation_unit="additions",
                operation_method="1 addition/output element",
            )

        # Semantic GELU
        if b in c_fc_outputs:
            gelu_inputs, gelu_outputs, gelu_saved = (
                capture_semantic_gelu(
                    model,
                    b,
                    c_fc_outputs[b],
                )
            )

            rows_by_operator[
                f"block{b}_gelu"
            ] = make_row(
                operator_name=f"block{b}_gelu",
                node_name=f"semantic_block{b}_gelu",
                fx_op="semantic",
                target="NewGELUActivation",
                operator_type="GELU",
                operator_role="MLP Activation",
                block=str(b),
                input_tensors=gelu_inputs,
                output_tensors=gelu_outputs,
                saved_tensors=gelu_saved,
                operation_count=float(
                    total_numel(gelu_outputs)
                ),
                operation_unit="elements",
                operation_method="processed output elements",
            )

        # MLP residual Add:
        # after c_proj and before next block / final LN
        next_boundary = len(ordered_names)

        if b < 3:
            next_ln1_name = f"block{b+1}_ln1"
            if next_ln1_name in rows_by_operator:
                next_node_name = rows_by_operator[
                    next_ln1_name
                ]["node_name"]
                next_boundary = order_index[next_node_name]
        else:
            if "final_ln" in rows_by_operator:
                next_boundary = order_index[
                    rows_by_operator["final_ln"]["node_name"]
                ]

        mlp_residual_region = ordered_names[
            anchors["c_proj"] + 1:
            next_boundary
        ]

        mlp_adds = [
            name
            for name in mlp_residual_region
            if is_add_node(
                inspector.records[name]["node"]
            )
        ]

        if mlp_adds:
            name = mlp_adds[-1]
            rec = inspector.records[name]
            node = rec["node"]

            rows_by_operator[
                f"block{b}_residual_mlp"
            ] = make_row(
                operator_name=f"block{b}_residual_mlp",
                node_name=node.name,
                fx_op=node.op,
                target=str(node.target),
                operator_type="Add",
                operator_role="MLP Residual",
                block=str(b),
                input_tensors=rec["input_tensors"],
                output_tensors=rec["output_tensors"],
                saved_tensors=rec["saved_tensors"],
                operation_count=float(
                    total_numel(rec["output_tensors"])
                ),
                operation_unit="additions",
                operation_method="1 addition/output element",
            )

    # --------------------------------------------------------
    # Exact 60-operator semantic order
    # --------------------------------------------------------

    semantic_order = [
        "embedding_token",
        "embedding_position",
    ]

    for b in range(4):
        semantic_order.extend([
            f"block{b}_ln1",
            f"block{b}_q_proj",
            f"block{b}_k_proj",
            f"block{b}_v_proj",
            f"block{b}_attn_qk_matmul",
            f"block{b}_attn_softmax",
            f"block{b}_attn_v_matmul",
            f"block{b}_out_proj",
            f"block{b}_residual_attn",
            f"block{b}_ln2",
            f"block{b}_mlp_c_fc",
            f"block{b}_gelu",
            f"block{b}_mlp_c_proj",
            f"block{b}_residual_mlp",
        ])

    semantic_order.extend([
        "final_ln",
        "lm_head",
    ])

    rows: List[Dict[str, Any]] = []

    missing = []

    for name in semantic_order:
        if name not in rows_by_operator:
            missing.append(name)
            continue

        row = rows_by_operator[name]

        row.update({
            "model": "TinyStories-33M",
            "parameters": total_params,
            "batch_size": args.batch_size,
            "image_size": "",
            "sequence_length": args.sequence_length,
        })

        rows.append(row)

    if missing:
        print()
        print("[WARNING] Missing semantic operators:")
        for name in missing:
            print(f"  - {name}")

    print(
        f"[SEMANTIC] captured {len(rows)}/60 operators"
    )

    return rows, fvcore_by_operator, unsupported


# ============================================================
# CSV / console output
# ============================================================

FIELDNAMES = [
    "model",
    "parameters",

    "operator",
    "node_name",
    "fx_op",
    "target",
    "operator_type",
    "operator_role",
    "block",

    "input_shape",
    "output_shape",

    "input_dtype",
    "output_dtype",

    "input_mb",
    "output_mb",

    "saved_tensor_count",
    "saved_tensor_shapes",
    "saved_tensor_dtypes",
    "saved_tensor_mb",

    "operation_count",
    "operation_count_m",
    "operation_unit",
    "operation_method",

    "batch_size",
    "image_size",
    "sequence_length",
]


def default_output_path(args) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    if args.model == "resnet18":
        name = (
            f"{timestamp}_"
            f"resnet18_"
            f"workload_"
            f"bs{args.batch_size}_"
            f"img{args.image_size}.csv"
        )
    else:
        name = (
            f"{timestamp}_"
            f"tinystories33m_"
            f"workload_"
            f"bs{args.batch_size}_"
            f"seq{args.sequence_length}.csv"
        )

    return (
        Path("exp_csv")
        / "workload"
        / args.model
        / name
    )


def write_csv(
    rows: List[Dict[str, Any]],
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in FIELDNAMES
            })


def print_rows(rows):
    print()
    print("=" * 145)
    print("OPERATOR WORKLOAD")
    print("=" * 145)

    print(
        f"{'operator':<38}"
        f"{'type':<15}"
        f"{'role':<25}"
        f"{'input MB':>10}"
        f"{'output MB':>11}"
        f"{'saved MB':>11}"
        f"{'count(M)':>12}"
        f"{'unit':>14}"
    )

    print("-" * 145)

    for row in rows:
        count = row["operation_count_m"]

        count_text = (
            f"{count:.3f}"
            if count is not None
            else "N/A"
        )

        print(
            f"{row['operator']:<38}"
            f"{row['operator_type']:<15}"
            f"{row['operator_role']:<25}"
            f"{row['input_mb']:>10.3f}"
            f"{row['output_mb']:>11.3f}"
            f"{row['saved_tensor_mb']:>11.3f}"
            f"{count_text:>12}"
            f"{row['operation_unit']:>14}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "resnet18",
            "tinystories",
        ],
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    # ResNet
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
    )

    # TinyStories
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    torch.manual_seed(1234)

    if args.model == "resnet18":
        rows, by_operator, unsupported = (
            profile_resnet18(args)
        )
    else:
        rows, by_operator, unsupported = (
            profile_tinystories(args)
        )

    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(args)
    )

    write_csv(
        rows,
        output_path,
    )

    print_rows(rows)

    print()
    print("=" * 145)
    print("FVCORE BY OPERATOR")
    print("=" * 145)

    if by_operator:
        for op, value in by_operator.items():
            print(
                f"{op:<35}"
                f"{value / 1e6:>12.3f} MFLOPs"
            )
    else:
        print("N/A")

    print()
    print("=" * 145)
    print("FVCORE UNSUPPORTED")
    print("=" * 145)
    print(unsupported)

    print()
    print(
        f"[OUTPUT] {output_path}"
    )


if __name__ == "__main__":
    main()
