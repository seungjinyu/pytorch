import os
import torch
import torch.nn.functional as F
import time
import csv

from splitmagic import SplitRuntime, ZMQServer
from splitmagic.utils.timing import CSVLogger
from splitmagic.runtime import read_dryrun_plan
from splitmagic.runtime import jin_set_payload_bytes_from_python

from splitmagic.runtime import is_causal_lm_model 
from splitmagic.runtime import run_model_forward

from contextlib import contextmanager

print(
    f"[Node B][SCRIPT_CHECK] "
    f"file={os.path.abspath(__file__)} "
    f"pid={os.getpid()} "
    f"torch={torch.__file__} "
    f"cuda={torch.cuda.is_available()}",
    flush=True,
)

@contextmanager
def nvtx_range(name: str):
    print(
        f"[NVTX][ENTER] {name}",
        flush=True,
    )

    pushed = False

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        pushed = True

    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()

        print(
            f"[NVTX][EXIT] {name}",
            flush=True,
        )

def append_recompute_experiment_csv(
    runtime,
    csv_path="./recompute_cost_experiments.csv",
):
    metrics = getattr(
        runtime,
        "last_experiment_metrics",
        None,
    )

    if not metrics:
        print(
            "[Node B][EXPERIMENT_CSV_SKIP] "
            "last_experiment_metrics is empty",
            flush=True,
        )
        return

    row = {
        "timestamp": time.time(),
        "model": "resnet18",
        "batch_size": int(
            os.environ.get("JIN_BATCH_SIZE", "32")
        ),

        "selection_policy": metrics.get(
            "node_a_selection_policy",
            "unknown",
        ),

        "drop_ratio": metrics.get(
            "node_a_drop_ratio",
            0.0,
        ),

        "saved_mb": metrics.get(
            "node_a_saved_mb",
            0.0,
        ),

        "dropped_count": metrics.get(
            "node_a_dropped_count",
            0,
        ),

        "network_mbps": float(
            os.environ.get(
                "JIN_NETWORK_MBPS",
                "500.0",
            )
        ),

        "run_id": int(
            os.environ.get("JIN_RUN_ID", "0")
        ),

        "missing_count": metrics.get(
            "missing_count",
            0,
        ),

        "recomputed_mb": metrics.get(
            "recomputed_mb",
            0.0,
        ),

        "node_a_predicted_operator_ms": metrics.get(
            "node_a_predicted_operator_ms",
            0.0,
        ),

        "node_a_predicted_recompute_ms": metrics.get(
            "node_a_predicted_recompute_ms",
            0.0,
        ),

        "node_a_predicted_inject_ms": metrics.get(
            "node_a_predicted_inject_ms",
            0.0,
        ),
        "executed_node_count": metrics.get(
            "recompute_executed_node_count",
            0,
        ),
        "profiled_node_count": metrics.get(
            "recompute_profiled_node_count",
            0,
        ),
        "missing_profile_count": metrics.get(
            "recompute_missing_profile_count",
            0,
        ),

        "estimated_grouped_ms": metrics.get(
            "estimated_grouped_ms",
            0.0,
        ),
        "predicted_operator_ms": metrics.get(
            "predicted_operator_ms",
            0.0,
        ),
        "actual_operator_ms": metrics.get(
            "actual_operator_ms",
            0.0,
        ),
        "actual_recompute_ms": metrics.get(
            "actual_recompute_ms",
            0.0,
        ),
        "recompute_wall_ms": metrics.get(
            "recompute_wall_ms",
            0.0,
        ),
        "recompute_overhead_ms": metrics.get(
            "recompute_overhead_ms",
            0.0,
        ),
        "inject_ms": metrics.get(
            "inject_ms",
            0.0,
        ),
        "torch_backward_ms": metrics.get(
            "torch_backward_ms",
            0.0,
        ),
        "backward_jin_ms": metrics.get(
            "backward_jin_ms",
            0.0,
        ),
    }

    # 실제 cost-model target
    row["actual_runtime_overhead_ms"] = max(
        0.0,
        row["recompute_wall_ms"]
        - row["actual_operator_ms"],
    )

    row["actual_total_recompute_cost_ms"] = (
        row["recompute_wall_ms"]
        + row["inject_ms"]
    )

    file_exists = os.path.exists(csv_path)
    file_empty = (
        not file_exists
        or os.path.getsize(csv_path) == 0
    )

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        if file_empty:
            writer.writeheader()

        writer.writerow(row)

    print(
        f"[Node B][EXPERIMENT_CSV_SAVE] "
        f"path={csv_path} "
        f"drop_ratio={row['drop_ratio']:.3f} "
        f"missing={row['missing_count']} "
        f"recomputed_mb={row['recomputed_mb']:.3f} "
        f"node_a_predicted_operator_ms="
        f"{row['node_a_predicted_operator_ms']:.3f} "
        f"predicted_operator_ms="
        f"{row['predicted_operator_ms']:.3f} "
        f"actual_operator_ms="
        f"{row['actual_operator_ms']:.3f} "
        f"actual_total_cost_ms="
        f"{row['actual_total_recompute_cost_ms']:.3f}",
        flush=True,
    )

def tensor_nbytes(t):
    return t.numel() * t.element_size()

def print_payload_size_summary(payload, topk=20):
    by_op = {}
    rows = []

    for k, t in payload.tensors.items():
        nbytes = tensor_nbytes(t)
        mb = nbytes / 1024 / 1024

        parts = k.split(":")
        if len(parts) >= 4 and parts[0] == "graph":
            op = parts[1]
            suffix = parts[3]
            group = f"{op}:{suffix}"
        else:
            group = k

        by_op[group] = by_op.get(group, 0) + nbytes
        rows.append((nbytes, k, tuple(t.shape), str(t.dtype)))

    total = sum(n for n, *_ in rows)

    print(
        f"[Node B][PAYLOAD_SIZE] total={total / 1024 / 1024:.3f} MB "
        f"num_keys={len(rows)}",
        flush=True,
    )

    print("[Node B][PAYLOAD_SIZE_BY_GROUP]", flush=True)
    for group, nbytes in sorted(by_op.items(), key=lambda x: x[1], reverse=True):
        print(
            f"  {group:24s} {nbytes / 1024 / 1024:10.3f} MB",
            flush=True,
        )

    print(f"[Node B][PAYLOAD_TOP{topk}]", flush=True)
    for nbytes, k, shape, dtype in sorted(rows, reverse=True)[:topk]:
        print(
            f"  {nbytes / 1024 / 1024:10.3f} MB  {k:35s} "
            f"shape={shape} dtype={dtype}",
            flush=True,
        )

def clone_grads(model):
    grads = {}
    grad_bytes = 0
    grad_tensors = 0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        g = p.grad.detach().cpu().clone()
        grads[name] = g

        grad_tensors += 1
        grad_bytes += g.numel() * g.element_size()

    return grads, grad_bytes, grad_tensors

def causal_lm_loss(logits, labels):
    """
    logits:
        [B, S, V]

    labels:
        [B, S]

    causal language modeling:
        token t로 token t+1 예측
    """

    if logits.ndim != 3:
        raise RuntimeError(
            "[CAUSAL_LM_LOSS] logits must be [B,S,V], "
            f"got {tuple(logits.shape)}"
        )

    if labels.ndim != 2:
        raise RuntimeError(
            "[CAUSAL_LM_LOSS] labels must be [B,S], "
            f"got {tuple(labels.shape)}"
        )

    shift_logits = (
        logits[:, :-1, :]
        .contiguous()
    )

    shift_labels = (
        labels[:, 1:]
        .contiguous()
    )

    return F.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.view(-1),
    )
def build_template_plan_on_b(
        model,
        batch_size,
        device,
        input_shape=(3, 32, 32),
        num_classes=10,
        sequence_length=9,
    ):

    template_plan_path = os.environ.get(
        "JIN_TEMPLATE_PLAN_PATH",
        "/tmp/jin_template_plan.tsv",
    )

    if os.path.exists(template_plan_path):
        os.remove(template_plan_path)

    os.environ["JIN_ROLE"] = "B"
    os.environ["JIN_DRYRUN"] = "1"
    os.environ["JIN_DRYRUN_PATH"] = (
        template_plan_path
    )

    model.zero_grad(
        set_to_none=True
    )

    causal_lm = is_causal_lm_model(
        model
    )

    # ========================================================
    # TinyStories / Causal LM
    # ========================================================

    if causal_lm:

        x_dummy = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(
                batch_size,
                sequence_length,
            ),
            dtype=torch.long,
            device=device,
        )

        # Framework 검증에서는
        # input_ids 자체를 causal-LM labels로 사용
        y_dummy = x_dummy.clone()

    # ========================================================
    # Existing CNN path
    # ========================================================

    else:

        input_shape = tuple(
            int(dim)
            for dim in input_shape
        )

        if len(input_shape) != 3:
            raise ValueError(
                "input_shape must be (C, H, W), "
                f"got {input_shape}"
            )

        x_dummy = torch.randn(
            batch_size,
            *input_shape,
            device=device,
        )

        y_dummy = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )

    # ========================================================
    # Dry-run backward
    # ========================================================

    with nvtx_range(
        "B_template_dryrun"
    ):

        out = run_model_forward(
            model,
            x_dummy,
        )

        # ----------------------------------------------------
        # TinyStories
        # ----------------------------------------------------

        if causal_lm:

            if out.ndim != 3:
                raise RuntimeError(
                    "[Node B] causal LM output "
                    "must be [B,S,V], "
                    f"got {tuple(out.shape)}"
                )

            if out.size(0) != batch_size:
                raise RuntimeError(
                    "[Node B] causal LM batch mismatch"
                )

            if out.size(1) != sequence_length:
                raise RuntimeError(
                    "[Node B] causal LM sequence mismatch: "
                    f"output={out.size(1)} "
                    f"expected={sequence_length}"
                )

            loss = causal_lm_loss(
                out,
                y_dummy,
            )

        # ----------------------------------------------------
        # Existing CNN
        # ----------------------------------------------------

        else:

            if out.ndim != 2:
                raise RuntimeError(
                    "[Node B] unexpected "
                    "model output shape: "
                    f"{tuple(out.shape)}"
                )

            if out.size(1) != num_classes:
                raise RuntimeError(
                    "[Node B] output class mismatch: "
                    f"model_output={out.size(1)}, "
                    f"num_classes={num_classes}"
                )

            loss = F.cross_entropy(
                out,
                y_dummy,
            )

        # 실제 backward dry-run
        loss.backward()

    model.zero_grad(
        set_to_none=True
    )

    os.environ.pop(
        "JIN_DRYRUN",
        None,
    )

    os.environ.pop(
        "JIN_DRYRUN_PATH",
        None,
    )

    plan = read_dryrun_plan(
        template_plan_path
    )

    if not plan:
        raise RuntimeError(
            "[Node B] template plan empty: "
            f"{template_plan_path}"
        )

    print(
        "[Node B][TEMPLATE_PLAN] "
        f"path={template_plan_path} "
        f"len={len(plan)} "
        f"batch_size={batch_size} "
        f"causal_lm={causal_lm} "
        f"sequence_length="
        f"{sequence_length if causal_lm else 'N/A'}",
        flush=True,
    )

    return plan

def write_execution_plan(plan, path=None):
    if path is None:
        path = os.environ.get(
            "JIN_EXECUTION_PLAN_PATH",
            "/tmp/jin_execution_plan.tsv",
        )

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "w") as f:
        for e in plan:
            f.write(
                f"{e['row_id']}\t"
                f"{e['op']}\t"
                f"{e['idx']}\t"
                f"{e['suffix']}\t"
                f"{e['shape']}\n"
            )

    os.environ["JIN_EXECUTION_PLAN_PATH"] = path

    print(
        f"[Node B][EXEC_PLAN_SAVE] "
        f"path={path} "
        f"len={len(plan)}",
        flush=True,
    )

    return path


def write_alias_tsv(aliases, path=None):

    if path is None:
        path = os.environ.get(
            "JIN_ALIAS_PATH",
            "/tmp/jin_payload_recv.bin.alias",
        )

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w") as f:
        for alias_key, canonical_key in aliases.items():
            f.write(f"{alias_key}\t{canonical_key}\n")

    print(
        f"[Node B][ALIAS_WRITE] path={path} n={len(aliases)}",
        flush=True,
    )

    return path

def run_node_b(
    model,
    endpoint="tcp://*:5555",
    csv_path="node_b_timing.csv",
    lr=0.1,
    template_batch_size=16,
    template_input_shape=(3, 32, 32),
    num_classes=10,
    log_level="2",
    send_grads=False,
):
    max_steps = int(
        os.environ.get("JIN_MAX_STEPS", "1")
    )

    print(
        f"[Node B][CONFIG] max_steps={max_steps}",
        flush=True,
    )
    
    printed_payload_summary = False

    # We are assuming the node B has a better computation power
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested_device = os.environ.get(
        "JIN_NODE_B_DEVICE",
        "cuda",
    ).lower()

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[Node B] CUDA was requested, "
                "but torch.cuda.is_available() is False"
            )

        device = torch.device("cuda")

    elif requested_device == "cpu":
        device = torch.device("cpu")

    else:
        raise ValueError(
            "[Node B] unsupported device: "
            f"{requested_device!r}"
        )

    model = model.to(device)

    causal_lm = is_causal_lm_model(
        model
    )

    model.train()

    if causal_lm:
        print(
            "[Node B][CAUSAL_LM_MODE] "
            "train mode with synchronized dropout RNG",
            flush=True,
        )

    os.environ["JIN_ROLE"] = "B"
    os.environ["JIN_LOG_LEVEL"] = log_level

    os.environ["JIN_BATCH_SIZE"] = str(
        template_batch_size
    )

    # ============================================================
    # TinyStories
    # ============================================================

    if causal_lm:

        template_sequence_length = int(
            os.environ.get(
                "JIN_SEQUENCE_LENGTH",
                "9",
            )
        )

        print(
            "[Node B][MODEL] "
            f"causal_lm=True "
            f"sequence_length="
            f"{template_sequence_length}",
            flush=True,
        )

    # ============================================================
    # Existing CNN
    # ============================================================

    else:

        template_input_shape = tuple(
            int(dim)
            for dim in template_input_shape
        )

        if len(template_input_shape) != 3:
            raise ValueError(
                "template_input_shape "
                "must be (C, H, W), "
                f"got {template_input_shape}"
            )

        template_sequence_length = 0


    # Build template plan
    template_plan = build_template_plan_on_b(
        model=model,
        batch_size=template_batch_size,
        device=device,
        input_shape=template_input_shape,
        num_classes=num_classes,
        sequence_length=template_sequence_length,
    )

    write_execution_plan(template_plan)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    runtime_b = SplitRuntime(model, role="B")

    logger = CSVLogger(
        csv_path,
        [
            "step",
            "loss",
            "payload_mb",
            "recv_wait_ms",
            "read_jin1_ms",
            "alias_ms",
            "state_load_ms",
            "backward_jin_ms",
            "clone_grads_ms",
            # "grads_mb",
            # "grad_tensors",
            "optimizer_step_ms",
            "state_dump_ms",
            "state_mb",
            "state_tensors",
            "send_reply_ms",
            "total_step_ms",
        ],
    )

    server = ZMQServer(endpoint)

    if causal_lm:
        print(
            "[Node B] listening "
            f"device={device} "
            f"model_family=causal_lm "
            f"batch_size={template_batch_size} "
            f"sequence_length={template_sequence_length} "
            f"vocab_size={model.config.vocab_size}",
            flush=True,
        )
    else:
        print(
            "[Node B] listening "
            f"device={device} "
            f"model_family=vision "
            f"batch_size={template_batch_size} "
            f"input_shape={template_input_shape} "
            f"num_classes={num_classes}",
            flush=True,
        )

    step = 0

    while True:

        # receive timer
        t_recv0 = time.perf_counter()

        with nvtx_range("B_wait_request"):
            req = server.recv_payload()

        t_recv1 = time.perf_counter()

        if req is None:
            break
        if isinstance(req, dict) and req.get("kind") == "get_template_plan":
            with nvtx_range("B_handle_template_plan_request"):
                server.send_reply({
                    "status": "ok",
                    "kind": "template_plan",
                    "template_plan": template_plan,
                })
                print(
                    f"[Node B][TEMPLATE_PLAN_SEND] len={len(template_plan)}",
                    flush=True,
                )
            continue
        with nvtx_range("B_handle_training_request"):
            t_step0 = time.perf_counter()

            # Read saved tensor from payload from Node A
            t_read0 = time.perf_counter()

            jin_payload = req["payload"]
            req["payload"] = jin_payload
            t_read1 = time.perf_counter()

            # Alias time 
            t_alias0 = time.perf_counter()
            with nvtx_range("B_alias_setup"):
                aliases = req.get("aliases", {})
                alias_path = os.environ.get(
                    "JIN_ALIAS_PATH",
                    req["payload_path"] + ".alias",
                )

                write_alias_tsv(
                    aliases,
                    alias_path,
                )

                req["payload"].meta = getattr(req["payload"], "meta", {})
                req["payload"].meta["aliases"] = aliases

                selection_meta = req.get("selection_meta", {})
                req["payload"].meta.update(selection_meta)

            print(
                "[Node B][SELECTION_META] "
                f"{selection_meta}",
                flush=True,
            )

            t_alias1 = time.perf_counter()

            print(
                f"[Node B][ALIAS] n={len(aliases)} "
                f"path={alias_path}",
                flush=True,
            )

            payload_keys = sorted(req["payload"].tensors.keys())

            if not printed_payload_summary:
                print_payload_size_summary(req["payload"], topk=30)
                printed_payload_summary = True

            t_state_load0 = time.perf_counter()

            with nvtx_range("B_load_state_dict"):
                
                if "state_dict" in req:
                    print("[Node B] state dict loaded\n")
                    model.load_state_dict(req["state_dict"])

            t_state_load1 = time.perf_counter()

            plan = req.get("dryrun_backward_plan", None)
            if not plan:
                plan = template_plan

            os.environ["JIN_ROLE"] = "B"
            os.environ["JIN_PAYLOAD_PATH"] = req["payload_path"]
            os.environ["JIN_STEP"] = str(step)

            with nvtx_range("B_prepare_inputs"):

                y = req["y"].to(device)

                request_batch_size = int(
                    req["batch_size"]
                )

                # ========================================================
                # TinyStories / Causal LM
                # ========================================================

                if causal_lm:

                    if y.ndim != 2:
                        raise RuntimeError(
                            "[Node B] causal LM labels "
                            "must be [B,S], "
                            f"got {tuple(y.shape)}"
                        )

                    if y.size(0) != request_batch_size:
                        raise RuntimeError(
                            "[Node B] batch-size mismatch: "
                            f"request={request_batch_size}, "
                            f"labels={y.size(0)}"
                        )

                    request_sequence_length = int(
                        req.get(
                            "sequence_length",
                            y.size(1),
                        )
                    )

                    if y.size(1) != request_sequence_length:
                        raise RuntimeError(
                            "[Node B] label/request sequence mismatch: "
                            f"labels={y.size(1)}, "
                            f"request={request_sequence_length}"
                        )

                    if (
                        request_sequence_length
                        != template_sequence_length
                    ):
                        raise RuntimeError(
                            "[Node B] sequence-length mismatch: "
                            f"request={request_sequence_length}, "
                            f"template={template_sequence_length}"
                        )

                    # IMPORTANT:
                    # Node A와 일부러 다른 input으로 forward.
                    # saved tensors는 JIN이 A 값으로 overwrite.
                    x_dummy = torch.randint(
                        low=0,
                        high=model.config.vocab_size,
                        size=(
                            request_batch_size,
                            request_sequence_length,
                        ),
                        dtype=torch.long,
                        device=device,
                    )

                    if y.numel() > 0:

                        min_label = int(
                            y.min().item()
                        )

                        max_label = int(
                            y.max().item()
                        )

                        if (
                            min_label < 0
                            or max_label
                            >= model.config.vocab_size
                        ):
                            raise RuntimeError(
                                "[Node B] LM token out of range: "
                                f"min={min_label}, "
                                f"max={max_label}, "
                                f"vocab="
                                f"{model.config.vocab_size}"
                            )

                # ========================================================
                # Existing CNN
                # ========================================================

                else:

                    request_input_shape = tuple(
                        int(dim)
                        for dim in req.get(
                            "input_shape",
                            template_input_shape,
                        )
                    )

                    if len(request_input_shape) != 3:
                        raise RuntimeError(
                            "[Node B] invalid request input shape: "
                            f"{request_input_shape}"
                        )

                    if (
                        request_input_shape
                        != template_input_shape
                    ):
                        raise RuntimeError(
                            "[Node B] request/template "
                            "input shape mismatch: "
                            f"request={request_input_shape}, "
                            f"template={template_input_shape}"
                        )

                    x_dummy = torch.randn(
                        request_batch_size,
                        *request_input_shape,
                        device=device,
                    )

                    if y.ndim != 1:
                        raise RuntimeError(
                            "[Node B] invalid label shape: "
                            f"{tuple(y.shape)}"
                        )

                    if (
                        y.size(0)
                        != request_batch_size
                    ):
                        raise RuntimeError(
                            "[Node B] batch-size mismatch: "
                            f"x={request_batch_size}, "
                            f"y={y.size(0)}"
                        )

                    if y.numel() > 0:

                        min_label = int(
                            y.min().item()
                        )

                        max_label = int(
                            y.max().item()
                        )

                        if (
                            min_label < 0
                            or max_label >= num_classes
                        ):
                            raise RuntimeError(
                                "[Node B] label out of range: "
                                f"min={min_label}, "
                                f"max={max_label}, "
                                f"num_classes={num_classes}"
                            )

            # Setting up to zero   
            optimizer.zero_grad(set_to_none=True)

            payload_keys = sorted(req["payload"].tensors.keys())
            print(
                f"[Node B][PAYLOAD] "
                f"num_keys={len(payload_keys)} "
                f"first={payload_keys[:10]}",
                flush=True,
            )

            t_backward0 = time.perf_counter()

            with nvtx_range("B_reserialize_payload_jin1"):
                payload_bytes = (
                    req["payload"].to_jin1_bytes()
                )

            with nvtx_range("B_set_jin_payload_bytes"):
                jin_set_payload_bytes_from_python(
                    payload_bytes=payload_bytes,
                    step=step,
                )

            if causal_lm:
                current_loss_fn = (
                    causal_lm_loss
                )
            else:
                current_loss_fn = (
                    F.cross_entropy
                )


            with nvtx_range("B_backward_jin_total"):
                dropout_seed = req.get(
                    "dropout_seed",
                    None,
                )

                if causal_lm and dropout_seed is None:
                    raise RuntimeError(
                        "[Node B] causal LM request "
                        "does not contain dropout_seed"
                    )

                loss = runtime_b.backward_jin(
                    x_dummy,
                    y=y,
                    payload=req["payload"],
                    loss_fn=current_loss_fn,
                    payload_path=req["payload_path"],
                    tensor_policy=req.get(
                        "tensor_policy",
                        None,
                    ),
                    dryrun_backward_plan=plan,
                    dropout_seed=dropout_seed,
                )
            if not causal_lm:
                append_recompute_experiment_csv(
                    runtime=runtime_b,
                    csv_path=os.environ.get(
                        "JIN_RECOMPUTE_EXPERIMENT_CSV",
                        "./recompute_cost_experiments.csv",
                    ),
                )

            experiment_metrics = getattr(
                runtime_b,
                "last_experiment_metrics",
                {},
            )

            t_backward1 = time.perf_counter()

            clone_grads_ms = 0.0

            if send_grads:
                t_grads0 = time.perf_counter()
                
                with nvtx_range("B_clone_grads_to_cpu"):
                    grads, grad_bytes, grad_tensors = (
                        clone_grads(model)
                    )
                t_grads1 = time.perf_counter()
                clone_grads_ms = (t_grads1 - t_grads0) * 1000

            else:
                grads = None 
                grad_bytes = 0 
                grad_tensors = 0

            t_opt0 = time.perf_counter()

            with nvtx_range("B_optimizer_step"):
                optimizer.step()

            t_opt1 = time.perf_counter()

            t_state_dump0 = time.perf_counter()


            with nvtx_range("B_state_dump_to_cpu"):
                updated_state = {}
                state_bytes = 0
                state_tensors = 0

                for k, v in model.state_dict().items():
                    t = v.detach().cpu().clone()
                    updated_state[k] = t

                    state_tensors += 1
                    state_bytes += t.numel() * t.element_size()

                t_state_dump1 = time.perf_counter()

            payload_mb = req["num_bytes"] / 1024 / 1024

            t_send0 = time.perf_counter()

            server_recv_complete_ts = req.get(
                "_server_recv_complete_ts",
                t_step0,
            )
            node_b_processing_ms = (
                t_send0 - server_recv_complete_ts
            ) * 1000

            with nvtx_range("B_build_reply"):

                reply = {
                    "status": "ok",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "bytes": req["num_bytes"],
                    "updated_state_dict": updated_state,

                    "missing_count": experiment_metrics.get(
                        "missing_count",
                        0,
                    ),

                    "estimated_grouped_ms": experiment_metrics.get(
                        "estimated_grouped_ms",
                        0.0,
                    ),

                    "predicted_operator_ms": experiment_metrics.get(
                        "predicted_operator_ms",
                        0.0,
                    ),

                    "recompute_wall_ms": experiment_metrics.get(
                        "recompute_wall_ms",
                        0.0,
                    ),

                    "recompute_overhead_ms": experiment_metrics.get(
                        "recompute_overhead_ms",
                        0.0,
                    ),

                    "recompute_prediction_ratio": experiment_metrics.get(
                        "recompute_prediction_ratio",
                        0.0,
                    ),

                    "recomputed_mb": experiment_metrics.get(
                        "recomputed_mb",
                        0.0,
                    ),

                    "recompute_executed_node_count": experiment_metrics.get(
                        "recompute_executed_node_count",
                        0,
                    ),

                    "recompute_profiled_node_count": experiment_metrics.get(
                        "recompute_profiled_node_count",
                        0,
                    ),

                    "recompute_missing_profile_count": experiment_metrics.get(
                        "recompute_missing_profile_count",
                        0,
                    ),

                    "recompute_plan_ms": experiment_metrics.get(
                        "recompute_plan_ms",
                        0.0,
                    ),

                    "inject_ms": experiment_metrics.get(
                        "inject_ms",
                        0.0,
                    ),

                    "torch_backward_ms": experiment_metrics.get(
                        "torch_backward_ms",
                        0.0,
                    ),

                    "backward_jin_ms": experiment_metrics.get(
                        "backward_jin_ms",
                        0.0,
                    ),

                    "node_b_processing_ms": node_b_processing_ms,
                }

            if send_grads:
                reply["grads"] = grads

            with nvtx_range("B_zmq_send_reply"):
                server.send_reply(reply)
                
            print(
                f"[Node B] step={step} "
                f"loss={loss.item():.6f} "
                f"payload_mb={payload_mb:.3f}"
            )

            t_send1 = time.perf_counter()

            t_step1 = time.perf_counter()

            recv_wait_ms = (t_recv1 - t_recv0) * 1000
            read_jin1_ms = (t_read1 - t_read0) * 1000
            alias_ms = (t_alias1 - t_alias0) * 1000
            state_load_ms = (t_state_load1 - t_state_load0) * 1000
            backward_jin_ms = (t_backward1 - t_backward0) * 1000
            optimizer_step_ms = (t_opt1 - t_opt0) * 1000
            state_dump_ms = (t_state_dump1 - t_state_dump0) * 1000
            send_reply_ms = (t_send1 - t_send0) * 1000
            total_step_ms = (t_step1 - t_step0) * 1000


            print(
                f"[Node B] step={step} "
                f"loss={loss.item():.6f} "
                f"payload_mb={payload_mb:.3f} "
                f"read_jin1_ms={read_jin1_ms:.3f} "
                f"backward_jin_ms={backward_jin_ms:.3f} "
                f"state_dump_ms={state_dump_ms:.3f} "
                f"state_mb={state_bytes / 1024 / 1024:.3f} "
                f"state_tensors={state_tensors} "
                f"send_reply_ms={send_reply_ms:.3f} "
                f"clone_grads_ms={clone_grads_ms:.3f} "
                # f"grads_mb={grad_bytes / 1024 / 1024:.3f} "
                # f"grad_tensors={grad_tensors} "
                f"total_step_ms={total_step_ms:.3f}",
                
                flush=True,
            )

            logger.write([
                step,
                float(loss.detach().cpu()),
                payload_mb,
                recv_wait_ms,
                read_jin1_ms,
                alias_ms,
                state_load_ms,
                backward_jin_ms,
                clone_grads_ms,
                # grad_bytes / 1024 / 1024,
                # grad_tensors,
                optimizer_step_ms,
                state_dump_ms,
                state_bytes / 1024 / 1024,
                state_tensors,
                send_reply_ms,
                total_step_ms,
            ])

            step += 1
            if step >= max_steps:
                print(
                    f"[Node B][DONE] "
                    f"completed_steps={step}",
                    flush=True,
                )
                break
