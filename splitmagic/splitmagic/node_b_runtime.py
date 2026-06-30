import os
import torch
import torch.nn.functional as F
import time

from splitmagic import SplitRuntime, ZMQServer
from splitmagic.utils.timing import CSVLogger
from splitmagic.runtime import read_dryrun_plan
# from splitmagic.resolver import read_jin1_payload
from splitmagic.runtime import jin_set_payload_bytes_from_python

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


def build_template_plan_on_b(model, batch_size, device):
    plan_path = "/tmp/jin_template_plan.tsv"

    if os.path.exists(plan_path):
        os.remove(plan_path)

    os.environ["JIN_ROLE"] = "B"
    os.environ["JIN_DRYRUN"] = "1"
    os.environ["JIN_DRYRUN_PATH"] = plan_path

    model.zero_grad(set_to_none=True)

    x_dummy = torch.randn(batch_size, 3, 32, 32, device=device)
    y_dummy = torch.zeros(batch_size, dtype=torch.long, device=device)

    out = model(x_dummy)
    loss = F.cross_entropy(out, y_dummy)
    loss.backward()

    model.zero_grad(set_to_none=True)

    os.environ.pop("JIN_DRYRUN", None)
    os.environ.pop("JIN_DRYRUN_PATH", None)

    plan = read_dryrun_plan(plan_path)

    if not plan:
        raise RuntimeError(f"[Node B] template plan empty: {plan_path}")

    print(f"[Node B][TEMPLATE_PLAN] path={plan_path} len={len(plan)}")

    return plan


def write_execution_plan(plan, path = "/tmp/jin_execution_plan.tsv"):
    plan_path = path

    with open(plan_path, "w") as f:
        for e in plan:
            f.write(
                f"{e['row_id']}\t"
                f"{e['op']}\t"
                f"{e['idx']}\t"
                f"{e['suffix']}\t"
                f"{e['shape']}\n"
            )

    os.environ["JIN_EXECUTION_PLAN_PATH"] = plan_path

    print(
        f"[Node B][EXEC_PLAN_SAVE]"
        f"path={plan_path}"
        f"len={len(plan)}",
        flush=True,
    )

    return plan_path


def write_alias_tsv(aliases, path="/tmp/jin_payload_alias.tsv"):
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
    log_level="2",
    send_grads=False,
):
    
    printed_payload_summary = False

    # We are assuming the node B has a better computation power
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = model.to(device)

    os.environ["JIN_ROLE"] = "B"
    os.environ["JIN_LOG_LEVEL"] = log_level

    # Build template plan 
    template_plan = build_template_plan_on_b(
        model=model,
        batch_size=template_batch_size,
        device=device,
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

    print("[Node B] listening")

    step = 0

    while True:

        # receive timer
        t_recv0 = time.perf_counter()

        req = server.recv_payload()

        t_recv1 = time.perf_counter()

        if req is None:
            break
        if isinstance(req, dict) and req.get("kind") == "get_template_plan":
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

        t_step0 = time.perf_counter()

        # Read saved tensor from payload from Node A
        t_read0 = time.perf_counter()

        jin_payload = req["payload"]
        req["payload"] = jin_payload
        t_read1 = time.perf_counter()

        # Alias time 
        t_alias0 = time.perf_counter()
        aliases = req.get("aliases", {})
        alias_path = req["payload_path"] + ".alias"

        # Key matching for duplicated values
        write_alias_tsv(aliases,alias_path)

        req["payload"].meta = getattr(req["payload"], "meta", {})
        req["payload"].meta["aliases"] = aliases
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

        y = req["y"].to(device)

        x_dummy = torch.randn(
            req["batch_size"],
            3,
            32,
            32,
            device=device,
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

        payload_bytes = req["payload"].to_jin1_bytes()
        jin_set_payload_bytes_from_python(
            payload_bytes=payload_bytes,
            step=step,
        )

        loss = runtime_b.backward_jin(
            x_dummy,
            y=y,
            payload=req["payload"],
            loss_fn=F.cross_entropy,
            payload_path=req["payload_path"],
            tensor_policy=req.get("tensor_policy", None),
            dryrun_backward_plan=plan
        )
        t_backward1 = time.perf_counter()

        if send_grads:
            t_grads0 = time.perf_counter()
            grads, grad_bytes, grad_tensors = clone_grads(model)
            t_grads1 = time.perf_counter()
        else:
            grads = None 
            grad_bytes = 0 
            grad_tensors = 0
        t_grads1 = time.perf_counter()

        t_opt0 = time.perf_counter()
        optimizer.step()
        t_opt1 = time.perf_counter()

        t_state_dump0 = time.perf_counter()

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

        reply = {
            "status": "ok",
            "step": step,
            "loss": float(loss.detach().cpu()),
            "bytes": req["num_bytes"],
            "updated_state_dict": updated_state,
        }

        if send_grads:
            reply["grads"] = grads

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
        clone_grads_ms = (t_grads1 - t_grads0) * 1000
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
