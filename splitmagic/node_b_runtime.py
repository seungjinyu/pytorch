import os
import torch
import torch.nn.functional as F

from splitmagic import SplitRuntime, ZMQServer
from splitmagic.utils.timing import CSVLogger
from splitmagic.runtime import read_dryrun_plan
from splitmagic.resolver import read_jin1_payload

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
    return {
        name: p.grad.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


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


def run_node_b(
    model,
    endpoint="tcp://*:5555",
    csv_path="node_b_timing.csv",
    lr=0.1,
    template_batch_size=32,
    log_level="4",
):
    
    printed_payload_summary = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.environ["JIN_ROLE"] = "B"
    os.environ["JIN_LOG_LEVEL"] = log_level

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
        ],
    )

    server = ZMQServer(endpoint)

    print("[Node B] listening")

    step = 0

    while True:
        req = server.recv_payload()

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

        jin_payload = read_jin1_payload(req["payload_path"])
        req["payload"] = jin_payload

        payload_keys = sorted(req["payload"].tensors.keys())
        print(
            f"[Node B][JIN1_PAYLOAD_RELOAD] "
            f"num_keys={len(payload_keys)} "
            f"first={payload_keys[:10]}",
            flush=True,
        )
        if not printed_payload_summary:
            print_payload_size_summary(req["payload"], topk=30)
            printed_payload_summary = True
        
        if "state_dict" in req:
            model.load_state_dict(req["state_dict"])

        plan = req.get("dryrun_backward_plan", None)
        if not plan:
            plan = template_plan

        # write_execution_plan(plan, step)

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

        optimizer.zero_grad(set_to_none=True)

        payload_keys = sorted(req["payload"].tensors.keys())
        print(
            f"[Node B][PAYLOAD] "
            f"num_keys={len(payload_keys)} "
            f"first={payload_keys[:10]}",
            flush=True,
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

        grads = clone_grads(model)

        optimizer.step()

        updated_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        payload_mb = req["num_bytes"] / 1024 / 1024

        server.send_reply({
            "status": "ok",
            "step": step,
            "loss": float(loss.detach().cpu()),
            "bytes": req["num_bytes"],
            "updated_state_dict": updated_state,
            "grads": grads,
        })

        print(
            f"[Node B] step={step} "
            f"loss={loss.item():.6f} "
            f"payload_mb={payload_mb:.3f}"
        )

        logger.write([
            step,
            float(loss.detach().cpu()),
            payload_mb,
        ])

        step += 1
