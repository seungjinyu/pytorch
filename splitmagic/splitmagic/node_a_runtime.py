import hashlib
import os
import time

import torch
import torch.nn.functional as F
from splitmagic import SplitRuntime, ZMQClient
from splitmagic.recompute_policy import RECOMPUTE_POLICIES
from splitmagic.utils.timing import CSVLogger


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

    print(
        f"[AUTO_DROP] ratio={drop_ratio} dropped={len(dropped)} saved_mb={saved / 1024 / 1024:.3f}",
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
    test_loader,
    endpoint="tcp://127.0.0.1:5555",
    csv_path="node_a_timing.csv",
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
):
    # environment setting
    os.environ["JIN_ROLE"] = "A"
    os.environ.pop("JIN_DRYRUN", None)
    os.environ.pop("JIN_DRYRUN_PATH", None)
    os.environ.pop("JIN_DRYRUN_TENSOR_DIR", None)

    # we are assuming node a is running on cpu.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # move model to cpu
    model = model.to(device)

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

    # initialize global_step and model train mode settings
    global_step = 0
    model.train()

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

    # Actual Training 
    for epoch in range(num_epochs):

        if global_step >= max_steps:
            break

        for _, (x, y) in enumerate(train_loader):
            if global_step >= max_steps:
                break
            x = x.to(device)
            y = y.to(device)

            iter_t0 = time.perf_counter()

            t0 = time.perf_counter()

            # 중요: A는 forward only. backward 호출 없음.
            payload = runtime_a.capture_jin_forward_plan(
                x=x,
                y=y,
                plan=plan,
            )
            payload.print_add_tensor_profile(
                prefix="[Payload][CAPTURE_ADD_TENSOR_PROFILE]"
            )

            t1 = time.perf_counter()
            capture_forward_ms = (t1 - t0) * 1000

            t0 = time.perf_counter()
            if enable_alias:
                payload = alias_duplicate_tensors(payload)

            t1 = time.perf_counter()
            alias_ms = (t1 - t0) * 1000
            
            t0 = time.perf_counter()

            if recompute_policy_name is not None:
                policy_conf = RECOMPUTE_POLICIES[recompute_policy_name]
            
                payload = auto_drop_by_ratio(
                    payload,
                    candidate_keys=policy_conf["drop"],
                    # protected_keys=policy_conf["keep"],
                    drop_ratio=auto_drop_ratio,
                )
            t1 = time.perf_counter()
            auto_drop_ms = (t1 - t0) * 1000

            policy_meta = payload.meta.get("tensor_policy", {})

            extra = {
                "tensor_policy": policy_meta,
                "dryrun_backward_plan": payload.meta.get("dryrun_backward_plan", []),
                "aliases": payload.meta.get("aliases", {}),
            }

            if global_step == 0:
                extra["state_dict"] = clone_state_dict(model)

            t_send0 = time.perf_counter()

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

            if grad_save_path is not None and "grads" in reply:
                torch.save(reply["grads"], grad_save_path)
                print(f"[Node A] saved grads to {grad_save_path}")

            t_load0 = time.perf_counter()

            if global_step == 0 and "grads" in reply:
                grad_keys = sorted(reply["grads"].keys())
                print(
                    f"[Node A][GRADS] num={len(grad_keys)} grads={grad_keys}",
                    flush=True,
                )
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

            global_step += 1

    # test_loss, test_acc = evaluate(model, test_loader,device=device)

    # temp skip
    # if grad_save_path is not None:
    #     print("The PATH was not set so saving int split_final.pt")
    #     torch.save(model.state_dict(), "split_final.pt")
    # else:
    #     print(f"The PATH for the gradient is {grad_save_path}")
    #     torch.save(model.state_dict(), grad_save_path)

    # print(
    #     f"[Eval] "
    #     f"test_loss={test_loss:.6f} "
    #     f"test_acc={test_acc * 100:.2f}%"
    # )

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
