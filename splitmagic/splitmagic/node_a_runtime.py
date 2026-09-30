import hashlib
import os
import time

import torch
import torch.nn.functional as F
from splitmagic import SplitRuntime, ZMQClient
from splitmagic.recompute_policy import RECOMPUTE_POLICIES
from splitmagic.cost_policy import (
    auto_drop_by_cost,
    load_recompute_cost_table, 
)
from splitmagic.utils.timing import CSVLogger
from contextlib import nullcontext, contextmanager
import torch

from splitmagic.runtime import is_causal_lm_model

@contextmanager
def nvtx_range_cpu(name: str):
    print(f"[NVTX-CPU][ENTER] {name}", flush=True)

    pushed = False

    try:
        torch.cuda.nvtx.range_push(name)
        pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()

        print(f"[NVTX-CPU][EXIT] {name}", flush=True)


def nvtx_range(name: str):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def tensor_fingerprint(t):
    tc = t.detach().cpu().contiguous()
    h = hashlib.sha256(tc.numpy().tobytes()).hexdigest()

    return (
        tuple(tc.shape),
        str(tc.dtype),
        h,
    )

def alias_duplicate_tensors(payload):
    """
    Generic tensor aliasing.

    If two payload tensors have exactly the same shape, dtype, and value,
    keep only one canonical tensor and replace the duplicate with an alias.

    This is model-agnostic and does not depend on ResNet/VGG row IDs.
    """
    if not hasattr(payload, "aliases"):
        payload.aliases = {}

    seen = {}
    removed = 0
    saved_bytes = 0

    for key, tensor in list(payload.tensors.items()):
        fp = tensor_fingerprint(tensor)

        if fp in seen:
            canonical_key = seen[fp]

            payload.aliases[key] = canonical_key
            payload.tensors.pop(key)

            nbytes = tensor.numel() * tensor.element_size()
            saved_bytes += nbytes
            removed += 1
        else:
            seen[fp] = key

    payload.meta["aliases"] = payload.aliases

    print(
        f"[ALIAS][SUMMARY] removed={removed} saved_mb={saved_bytes / 1024 / 1024:.3f}",
        flush=True,
    )

    return payload


def auto_drop_for_recompute_probe(payload, drop_keys=None):
    if drop_keys is None:
        drop_keys = set()

    dropped = []
    dropped_bytes = 0

    for key in sorted(drop_keys):
        tensor = payload.tensors.pop(key, None)

        if tensor is None:
            print(f"[DROP_PROBE_SKIP] missing key={key}", flush=True)
            continue

        nbytes = tensor.numel() * tensor.element_size()
        dropped.append(key)
        dropped_bytes += nbytes

        print(
            f"[DROP_PROBE] key={key} saved_mb={nbytes / 1024 / 1024:.3f}",
            flush=True,
        )

    payload.meta["drop_probe_keys"] = dropped

    print(
        f"[DROP_PROBE_SUMMARY] dropped={len(dropped)} saved_mb={dropped_bytes / 1024 / 1024:.3f}",
        flush=True,
    )

    return payload


def auto_drop_by_ratio(
    payload,
    candidate_keys,
    # protected_keys=None,
    drop_ratio=0.5,
):
    rows = []
    # protected_keys = protected_keys or set()

    for key in candidate_keys:
        t = payload.tensors.get(key)
        if t is None:
            continue

        nbytes = t.numel() * t.element_size()
        rows.append((nbytes, key))

    total_bytes = sum(t.numel() * t.element_size() for t in payload.tensors.values())

    target = int(total_bytes * drop_ratio)

    rows.sort(reverse=True)

    dropped = []
    saved = 0

    for nbytes, key in rows:
        if saved >= target:
            break

        payload.tensors.pop(key, None)
        dropped.append(key)
        saved += nbytes

    payload.meta["auto_dropped_keys"] = dropped
    payload.meta["drop_ratio"] = float(drop_ratio)
    payload.meta["dropped_count"] = len(dropped)
    payload.meta["saved_mb"] = saved / 1024 / 1024

    print(
        f"[AUTO_DROP] ratio={drop_ratio} "
        f"dropped={len(dropped)} "
        f"saved_mb={saved / 1024 / 1024:.3f}",
        flush=True,
    )

    return payload


def drop_payload_keys(payload, drop_keys=None):
    """
    Drop selected payload tensors by key.

    This is intentionally config-driven.
    The runtime should not hard-code model-specific keys such as
    ResNet18 BN/ReLU row IDs.
    """
    if not drop_keys:
        payload.meta["dropped_keys"] = []
        return payload

    drop_keys = set(drop_keys)
    removed = 0
    saved_bytes = 0

    for key in drop_keys:
        tensor = payload.tensors.pop(key, None)

        if tensor is not None:
            removed += 1
            saved_bytes += tensor.numel() * tensor.element_size()

    payload.meta["dropped_keys"] = sorted(drop_keys)

    print(
        f"[DROP][SUMMARY] removed={removed} saved_mb={saved_bytes / 1024 / 1024:.3f}",
        flush=True,
    )

    return payload



def run_node_a(
    model,
    train_loader,
    # test_loader=None,
    endpoint="tcp://127.0.0.1:5555",
    csv_path="node_a_timing.csv",
    experiment_csv_path="drop_ratio_experiment_100mb.csv",
    num_epochs=10,
    max_steps=60000,
    policy="full",
    optional_keys=None,
    grad_save_path=None,
    key_mode="module_debug",
    dryrun_plan=False,
    template_plan_path="/tmp/jin_template_plan_a.tsv",
    auto_drop_ratio=0.5,
    enable_alias=True,
    recompute_policy_name=None,

    selection_policy="ratio",
    recompute_cost_csv=None,
    network_mbps=None,
    inject_ms_per_mb=0.0,
    min_benefit_ms=0.0,
    max_cost_drop_ratio=None,
):
    # environment setting
    os.environ["JIN_ROLE"] = "A"
    os.environ.pop("JIN_DRYRUN", None)
    os.environ.pop("JIN_DRYRUN_PATH", None)
    os.environ.pop("JIN_DRYRUN_TENSOR_DIR", None)

    run_id = int(
        os.environ.get("JIN_EXPERIMENT_RUN_ID", "0")
    )

    # we are assuming node a is running on cpu.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # move model to cpu
    model = model.to(device)

    causal_lm = is_causal_lm_model(model)

    print(
        f"[Node A][MODEL] "
        f"causal_lm={causal_lm} "
        f"model_type={getattr(getattr(model, 'config', None), 'model_type', None)}",
        flush=True,
    )

    # if policy is not "full", print a warning
    if policy != "full":
        print(f"[Node A][WARN] policy argument is currently unused: {policy}")
    # if optional_keys is not None, print a warning
    if optional_keys is not None:
        print(f"[Node A][WARN] optional_keys argument is currently unused")
    # if key_mode is not "module_debug", print a warning
    if key_mode != "module_debug":
        print(f"[Node A][WARN] key_mode argument is currently unused: {key_mode}")

    # Split Runtime for Node A
    runtime_a = SplitRuntime(model, role="A")

    # Split ZMQ Client for Node A
    client = ZMQClient(endpoint)

    # CSV Logger for Node A
    logger = CSVLogger(
        csv_path,
        [
            "step",
            "loss",
            "payload_mb",
            "capture_forward_ms",
            "alias_ms",
            "auto_drop_ms",
            "send_recv_ms",
            "state_load_ms",
            "total_ms",
        ],
    )
    experiment_logger = CSVLogger(
        experiment_csv_path,
        [
            "run_id",
            "network_mbps",
            "drop_ratio",
            "payload_mb",
            "saved_mb",
            "dropped_count",
            "missing_count",

            "estimated_grouped_ms",
            "predicted_operator_ms",

            "recompute_wall_ms",
            "recompute_overhead_ms",
            "recompute_prediction_ratio",
            "recomputed_mb",
            "inject_ms",
            "total_recompute_cost_ms",

            "recompute_executed_node_count",
            "recompute_profiled_node_count",
            "recompute_missing_profile_count",

            "recompute_plan_ms",
            "torch_backward_ms",
            "backward_jin_ms",

            "request_round_trip_ms",
            "application_communication_ms",
            "client_send_pyobj_ms",
            "client_recv_pyobj_ms",
            "node_b_processing_ms",
            "node_a_total_ms",
            "loss",
        ],
        append=True,
    )
    # initialize global_step and model train mode settings
    global_step = 0

    model.train()

    if causal_lm:
        print(
            "[Node A][CAUSAL_LM_MODE] "
            "train mode with synchronized dropout RNG",
            flush=True,
        )

    if selection_policy not in {"ratio","cost","none"}:
        raise ValueError(
            f"Unsupported selection_policy={selection_policy!r}."
            "Expected one of: ratio, cost, none"
        )
    
    recompute_cost_table = None

    if selection_policy == "cost":
        if recompute_cost_csv is None:
            raise ValueError(
                "recompute_cost_csv is required "
                "when selection_policy='cost'"
            )

        if network_mbps is None or network_mbps <= 0:
            raise ValueError(
                "A positive network_mbps is required "
                "when selection_policy='cost'"
            )

        recompute_cost_table = load_recompute_cost_table(
            recompute_cost_csv
        )

        print(
            f"[Node A][COST_POLICY_LOAD] "
            f"path={recompute_cost_csv} "
            f"entries={len(recompute_cost_table)} "
            f"network_mbps={network_mbps}",
            flush=True,
        )

        print(
            f"[Node A][COST_MODEL] "
            f"inject_ms_per_mb={inject_ms_per_mb:.6f}",
            flush=True,
        )



    #  raise error if dryrun_plan is False
    if not dryrun_plan:
        raise RuntimeError(
            "[Node A] dryrun_plan=False path is disabled. Use dryrun_plan=True with B-generated template plan."
        )

    # request template plan from Node B
    plan = client.request_template_plan()

    if not plan:
        raise RuntimeError(f"[Node A] template plan is empty or missing: {template_plan_path}")
    with open(template_plan_path, "w") as f:
        for e in plan:
            f.write(f"{e['row_id']}\t{e['op']}\t{e['idx']}\t{e['suffix']}\t{e['shape']}\n")

    print(
        f"[Node A][TEMPLATE_PLAN_LOAD] path={template_plan_path} len={len(plan)}",
        flush=True,
    )
    max_steps = int(os.environ.get("JIN_MAX_STEPS", "1"))
    completed_steps = 0

    # Actual Training 
    for epoch in range(num_epochs):

        if global_step >= max_steps:
            break

        for _, (x, y) in enumerate(train_loader):
            if global_step >= max_steps:
                break
            x = x.to(device)
            y = y.to(device)

            with nvtx_range_cpu(f"A_step_{global_step}"):

                iter_t0 = time.perf_counter()

                t0 = time.perf_counter()

                with nvtx_range_cpu("A_capture_forward"):

                    # TinyStories는 아직 FX profiling/recompute 경로를
                    # 연결하지 않았으므로 Phase 1에서는 skip.
                    if (
                        global_step == 0
                        and not causal_lm
                    ):
                        
                        runtime_a.profile_forward_layers(
                            x,
                            csv_path="./forward_layer_profile_cpu.csv",
                        )
                    dropout_seed = None
                    if causal_lm:
                        dropout_seed = 1234 + global_step

                        torch.manual_seed(
                            dropout_seed
                        )

                        print(
                            f"[Node A][DROPOUT_SEED] "
                            f"step={global_step} "
                            f"seed={dropout_seed}",
                            flush=True,
                        )

                    # 중요: A는 forward only. backward 호출 없음.
                    payload = runtime_a.capture_jin_forward_plan(
                        x=x,
                        plan=plan,
                    )

                with nvtx_range_cpu("A_payload_profile_print"):

                    payload.print_add_tensor_profile(
                        prefix="[Payload][CAPTURE_ADD_TENSOR_PROFILE]"
                    )

                t1 = time.perf_counter()
                capture_forward_ms = (t1 - t0) * 1000

                t0 = time.perf_counter()
                with nvtx_range_cpu("A_alias_duplicate"):
                    if (
                        enable_alias
                        and not is_causal_lm_model(model)
                    ):
                        payload = alias_duplicate_tensors(
                            payload
                        )

                t1 = time.perf_counter()
                alias_ms = (t1 - t0) * 1000
                
                t0 = time.perf_counter()

                with nvtx_range_cpu(
                    f"A_selection_{selection_policy}"
                ):

                    # --------------------------------------------------------
                    # TinyStories Phase 1:
                    # FULL payload only.
                    # No drop / no recompute selection.
                    # --------------------------------------------------------
                    if causal_lm:

                        payload.meta["selection_policy"] = "none"
                        payload.meta["auto_dropped_keys"] = []
                        payload.meta["drop_ratio"] = 0.0
                        payload.meta["dropped_count"] = 0
                        payload.meta["saved_mb"] = 0.0

                        print(
                            "[Node A][SELECTION_STAGE] "
                            "causal_lm=True -> FULL PAYLOAD",
                            flush=True,
                        )

                    # --------------------------------------------------------
                    # Existing CNN path
                    # --------------------------------------------------------
                    elif recompute_policy_name is not None:

                        policy_conf = RECOMPUTE_POLICIES[
                            recompute_policy_name
                        ]

                        candidate_keys = policy_conf["drop"]

                        print(
                            f"[Node A][SELECTION_STAGE] "
                            f"policy={selection_policy}",
                            flush=True,
                        )

                        if selection_policy == "ratio":

                            payload = auto_drop_by_ratio(
                                payload,
                                candidate_keys=candidate_keys,
                                drop_ratio=auto_drop_ratio,
                            )

                            payload.meta[
                                "selection_policy"
                            ] = "ratio"

                        elif selection_policy == "cost":

                            print(
                                "[Node A][ENTER_COST]",
                                flush=True,
                            )

                            cost0 = time.perf_counter()

                            payload = auto_drop_by_cost(
                                payload=payload,
                                candidate_keys=candidate_keys,
                                cost_table=recompute_cost_table,
                                network_mbps=network_mbps,
                                min_benefit_ms=min_benefit_ms,
                                max_drop_ratio=max_cost_drop_ratio,
                            )

                            cost1 = time.perf_counter()

                            print(
                                "[AUTO_DROP_COST_TABLE_TIME] "
                                f"{(cost1 - cost0) * 1000:.3f} ms",
                                flush=True,
                            )

                        elif selection_policy == "none":

                            payload.meta["selection_policy"] = "none"
                            payload.meta["auto_dropped_keys"] = []
                            payload.meta["drop_ratio"] = 0.0
                            payload.meta["dropped_count"] = 0
                            payload.meta["saved_mb"] = 0.0

                        print(
                            "[Node A][SELECTION_END]",
                            flush=True,
                        )

                t1 = time.perf_counter()
                auto_drop_ms = (t1 - t0) * 1000

                policy_meta = payload.meta.get("tensor_policy", {})

                extra = {
                    "tensor_policy": policy_meta,

                    "dryrun_backward_plan": payload.meta.get(
                        "dryrun_backward_plan",
                        [],
                    ),

                    "aliases": payload.meta.get(
                        "aliases",
                        {},
                    ),

                    # Cost-policy metadata
                    "selection_meta": {
                        "selection_policy": payload.meta.get(
                            "selection_policy",
                            "none",
                        ),

                        "drop_ratio": payload.meta.get(
                            "drop_ratio",
                            0.0,
                        ),

                        "saved_mb": payload.meta.get(
                            "saved_mb",
                            0.0,
                        ),

                        "dropped_count": payload.meta.get(
                            "dropped_count",
                            0,
                        ),

                        "predicted_operator_ms": payload.meta.get(
                            "predicted_operator_ms",
                            0.0,
                        ),

                        "predicted_recompute_ms": payload.meta.get(
                            "predicted_recompute_ms",
                            0.0,
                        ),

                        "predicted_inject_ms": payload.meta.get(
                            "predicted_inject_ms",
                            0.0,
                        ),

                        "predicted_send_saved_ms": payload.meta.get(
                            "predicted_send_saved_ms",
                            0.0,
                        ),

                        "predicted_benefit_ms": payload.meta.get(
                            "predicted_benefit_ms",
                            0.0,
                        ),
                    },
                }


                # ------------------------------------------------------------
                # Model-specific request metadata
                # ------------------------------------------------------------

                if causal_lm:

                    extra["model_family"] = (
                        "causal_lm"
                    )

                    extra["sequence_length"] = int(
                        x.size(1)
                    )

                    # A forward에서 실제 사용한 dropout RNG seed
                    extra["dropout_seed"] = int(
                        dropout_seed
                    )

                else:

                    extra["model_family"] = (
                        "vision"
                    )

                    extra["input_shape"] = tuple(
                        int(v)
                        for v in x.shape[1:]
                    )

                if global_step == 0:
                    extra["state_dict"] = clone_state_dict(model)

                t_send0 = time.perf_counter()

                with nvtx_range_cpu("A_request_reply_total"):

                    reply = client.send_payload(
                        payload=payload,
                        y=y,
                        batch_size=x.size(0),
                        extra=extra,
                    )

                t_send1 = time.perf_counter()

                if reply["status"] != "ok":
                    print("[Node A] bad reply:", reply)
                    break

                payload_mb = reply["bytes"] / 1024 / 1024

                drop_ratio_value = float(
                    payload.meta.get("drop_ratio", 0.0)
                )

                saved_mb = float(
                    payload.meta.get("saved_mb", 0.0)
                )

                dropped_count = int(
                    payload.meta.get("dropped_count", 0)
                )

                if grad_save_path is not None and "grads" in reply:
                    torch.save(reply["grads"], grad_save_path)
                    print(f"[Node A] saved grads to {grad_save_path}")

                t_load0 = time.perf_counter()

                if global_step == 0 and "grads" in reply:
                    grad_keys = sorted(reply["grads"].keys())

                with nvtx_range_cpu("A_load_updated_state"):
                    model.load_state_dict(reply["updated_state_dict"])

                t_load1 = time.perf_counter()
                send_recv_ms = (t_send1 - t_send0) * 1000
                state_load_ms = (t_load1 - t_load0) * 1000

                total_ms = (
                    capture_forward_ms
                    + alias_ms
                    + auto_drop_ms
                    + send_recv_ms
                    + state_load_ms
                )
                iteration_wall_ms = (time.perf_counter() - iter_t0) * 1000

                print(
                    f"[Node A] epoch={epoch} step={global_step} "
                    f"loss={reply['loss']:.6f} "
                    f"payload_mb={payload_mb:.3f} "
                    f"capture_forward_ms={capture_forward_ms:.3f} "
                    f"alias_ms={alias_ms:.3f} "
                    f"auto_drop_ms={auto_drop_ms:.3f} "
                    f"send_recv_ms={send_recv_ms:.3f} "
                    f"application_communication_ms="
                    f"{reply.get('application_communication_ms', 0.0):.3f} "
                    f"state_load_ms={state_load_ms:.3f} "
                    f"total_ms={total_ms:.3f}",
                    flush=True,
                )

                logger.write(
                    [
                        global_step,
                        reply["loss"],
                        payload_mb,
                        capture_forward_ms,
                        alias_ms,
                        auto_drop_ms,
                        send_recv_ms,
                        state_load_ms,
                        total_ms,
                    ]
                )
                recompute_wall_ms_value = float(
                    reply.get("recompute_wall_ms", 0.0)
                )

                inject_ms_value = float(
                    reply.get("inject_ms", 0.0)
                )

                total_recompute_cost_ms = (
                    recompute_wall_ms_value
                    + inject_ms_value
                )
                experiment_logger.write([
                    run_id,
                    network_mbps,
                    drop_ratio_value,
                    payload_mb,
                    saved_mb,
                    dropped_count,
                    reply.get("missing_count", 0),

                    reply.get("estimated_grouped_ms", 0.0),
                    reply.get("predicted_operator_ms", 0.0),

                    recompute_wall_ms_value,
                    reply.get("recompute_overhead_ms", 0.0),
                    reply.get("recompute_prediction_ratio", 0.0),
                    reply.get("recomputed_mb", 0.0),
                    inject_ms_value,
                    total_recompute_cost_ms,

                    reply.get(
                        "recompute_executed_node_count",
                        0,
                    ),
                    reply.get(
                        "recompute_profiled_node_count",
                        0,
                    ),
                    reply.get(
                        "recompute_missing_profile_count",
                        0,
                    ),

                    reply.get("recompute_plan_ms", 0.0),
                    reply.get("torch_backward_ms", 0.0),
                    reply.get("backward_jin_ms", 0.0),

                    reply.get("request_round_trip_ms", send_recv_ms),
                    reply.get("application_communication_ms", 0.0),
                    reply.get("client_send_pyobj_ms", 0.0),
                    reply.get("client_recv_pyobj_ms", 0.0),
                    reply.get("node_b_processing_ms", 0.0),
                    total_ms,
                    reply["loss"],
                ])

                global_step += 1
                completed_steps += 1

                if completed_steps >= max_steps:
                    print(
                        f"[Node A][DONE] completed_steps={completed_steps}",
                        flush=True,
                    )
                    return

    print("[Node A] done")


@torch.no_grad()
def evaluate(model, test_loader, device="cpu"):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = F.cross_entropy(out, y)

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)

    model.train()
    return total_loss / total, correct / total
