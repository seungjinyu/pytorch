import os
import time
import torch
import torch.nn.functional as F
import json

from splitmagic import SplitRuntime, ZMQServer
from splitmagic.utils.timing import CSVLogger

def clone_grads(model):
    return {
        name: p.grad.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

def run_node_b(
    model,
    endpoint="tcp://*:5555",
    csv_path="node_b_timing.csv",
    lr=0.1,
):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = model.to(device)

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

        dryrun_backward_plan = req.get("dryrun_backward_plan", [])
        
        if dryrun_backward_plan:

            plan_path = "/tmp/jin_execution_plan.tsv"

            with open(plan_path,"w" ) as f :
                for e in dryrun_backward_plan:
                    f.write(f"{e['row_id']}\t{e['op']}\t{e['idx']}\t{e['suffix']}\t{e['shape']}\n")

            os.environ["JIN_EXECUTION_PLAN_PATH"] = plan_path

            print(
                f"[Node B][EXEC_PLAN_SAVE]"
                f"path={plan_path}"
                f"len={len(dryrun_backward_plan)}",
                flush=True
            )

        else :
            os.environ.pop("JIN_EXECUTION_PLAN_PATH",None)
            print("[Node B][EXEC_PLAN_EMPTY]")

        if req is None:
            break

        if "state_dict" in req:
            model.load_state_dict(req["state_dict"])

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

        print("[Node B][PAYLOAD_KEYS]", sorted(req["payload"].tensors.keys()))

        # t = read_jin1_tensor(req["payload_path"], "relu:1:out")
        # print("[TEST][JIN1_READ] relu:1:out", t.shape, t.dtype, t.mean().item())

        loss = runtime_b.backward_jin(
            x_dummy,
            y=y,
            payload=req["payload"],
            loss_fn=F.cross_entropy,
            payload_path=req["payload_path"],
            tensor_policy=req.get("tensor_policy",None),
        )

        # torch._C._jin_dump_used_keys("/tmp/jin_used_keys.txt")
        # 비교를 위한 grad 
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
            # 비교를 위한 추가 이후 제거 예정
            "grads": grads,
        })

        print(
            f"[Node B] step={step} "
            f"loss={loss.item():.6f} "
            f"payload_mb={payload_mb:.3f} "
        )

        logger.write([
            step,
            float(loss.detach().cpu()),
            payload_mb,
        ])

        step += 1