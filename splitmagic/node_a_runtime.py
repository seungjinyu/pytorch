import time
import torch
import torch.nn.functional as F
import os

from splitmagic import SplitRuntime, ZMQClient
from splitmagic.utils.timing import CSVLogger
from splitmagic.runtime import read_dryrun_plan


def clone_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


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
):
    os.environ["JIN_ROLE"] = "A"
    os.environ.pop("JIN_DRYRUN", None)
    os.environ.pop("JIN_DRYRUN_PATH", None)
    os.environ.pop("JIN_DRYRUN_TENSOR_DIR", None)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)

    if policy != "full":
        print(f"[Node A][WARN] policy argument is currently unused: {policy}")

    if optional_keys is not None:
        print(f"[Node A][WARN] optional_keys argument is currently unused")

    if key_mode != "module_debug":
        print(f"[Node A][WARN] key_mode argument is currently unused: {key_mode}")

    runtime_a = SplitRuntime(model, role="A")
    client = ZMQClient(endpoint)

    logger = CSVLogger(
        csv_path,
        [
            "step",
            "loss",
            "payload_mb",
            "capture_ms",
            "send_recv_ms",
            "state_load_ms",
            "total_ms",
        ],
    )

    global_step = 0
    model.train()

    if not dryrun_plan:
        raise RuntimeError(
            "[Node A] dryrun_plan=False path is disabled. "
            "Use dryrun_plan=True with B-generated template plan."
        )
    # plan = read_dryrun_plan(template_plan_path)

    plan = client.request_template_plan()

    if not plan:
        raise RuntimeError(
            f"[Node A] template plan is empty or missing: {template_plan_path}"
        )
    with open(template_plan_path,"w") as f:
        for e in plan:
            f.write(
                f"{e['row_id']}\t"
                f"{e['op']}\t"
                f"{e['idx']}\t"
                f"{e['suffix']}\t"
                f"{e['shape']}\n"
            )
    
    print(
        f"[Node A][TEMPLATE_PLAN_LOAD] "
        f"path={template_plan_path} "
        f"len={len(plan)}",
        flush=True,
    )
    
    for epoch in range(num_epochs):

        if global_step >= max_steps:
            break

        for _, (x, y) in enumerate(train_loader):
            if global_step >= max_steps:
                break
            x = x.to(device)
            y = y.to(device)

            t0 = time.perf_counter()
            t_capture0 = time.perf_counter()

            # 중요: A는 forward only. backward 호출 없음.
            payload = runtime_a.capture_jin_forward_plan(
                x=x,
                y=y,
                plan=plan,
            )
            # drop_keys = {
            #     "graph:relu:7:result",
            #     "graph:relu:6:result",
            #     "graph:relu:5:result",
            #     "graph:relu:4:result",
            #     "graph:relu:3:result",
            #     "graph:relu:2:result",
            #     "graph:relu:1:result",
            #     "graph:conv:3:input",
            #     "graph:conv:1:input",
            #     "graph:conv:0:input",
            #     "graph:bn:7:input",
            #     "graph:bn:6:input",
            #     "graph:bn:5:input",
            #     "graph:bn:4:input",
            #     "graph:bn:3:input",
            #     "graph:bn:2:input",
            #     "graph:bn:1:input",
            #     # "graph:conv:7:input",
            #     # "graph:maxpool2d:4:input",
            #     # "graph:maxpool2d:4:indices",
            # }
            # drop_keys = {
            #     "graph:addmm:0:mat1",
            #     "graph:conv:18:input",
            #     "graph:conv:17:input",
            #     "graph:conv:16:input",
            #     "graph:conv:15:input",
            #     "graph:conv:14:input",
            #     "graph:conv:13:input",
            #     "graph:conv:12:input",
            #     "graph:conv:11:input",
            #     "graph:conv:10:input",
            #     "graph:conv:9:input",
            #     "graph:conv:8:input",
            #     "graph:bn:19:input",
            #     "graph:bn:18:input",
            #     "graph:bn:17:input",
            #     "graph:bn:16:input",
            #     "graph:bn:15:input",
            #     "graph:bn:14:input",
            #     "graph:bn:13:input",
            #     "graph:bn:12:input",
            # }

            # for k in drop_keys:
            #     payload.tensors.pop(k, None)

            policy_meta = payload.meta.get("tensor_policy", {})

            t_capture1 = time.perf_counter()

            extra = {
                "tensor_policy": policy_meta,
                "dryrun_backward_plan": payload.meta.get("dryrun_backward_plan", []),
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

            # if grad_save_path is not None and "grads" in reply:
            #     torch.save(reply["grads"], grad_save_path)
            #     print(f"[Node A] saved grads to {grad_save_path}")

                # print(
                #     f"[Node A] epoch={epoch} "
                #     f"step={global_step} "
                #     f"loss={reply['loss']:.6f} "
                #     f"payload_mb={payload_mb:.3f} "
                #     # f"capture_ms={capture_ms:.2f} "
                #     # f"send_recv_ms={send_recv_ms:.2f} "
                #     # f"state_load_ms={state_load_ms:.2f} "
                #     # f"total_ms={total_ms:.2f}",
                #     ,
                #     flush=True
                # )

                # return

            t_load0 = time.perf_counter()

            if global_step == 0 and "grads" in reply:
                grad_keys = sorted(reply["grads"].keys())
                print(
                    f"[Node A][GRADS] "
                    f"num={len(grad_keys)} "
                    f"grads={grad_keys}",
                    flush=True,
                )
            model.load_state_dict(reply["updated_state_dict"])

            t_load1 = time.perf_counter()

            total_ms = (time.perf_counter() - t0) * 1000
            capture_ms = (t_capture1 - t_capture0) * 1000
            send_recv_ms = (t_send1 - t_send0) * 1000
            state_load_ms = (t_load1 - t_load0) * 1000
            # payload_mb = reply["bytes"] / 1024 / 1024

            print(
                f"[Node A] epoch={epoch} "
                f"step={global_step} "
                f"loss={reply['loss']:.6f} "
                f"payload_mb={payload_mb:.3f} "
                # f"capture_ms={capture_ms:.2f} "
                # f"send_recv_ms={send_recv_ms:.2f} "
                # f"state_load_ms={state_load_ms:.2f} "
                # f"total_ms={total_ms:.2f}",
                ,
                flush=True
            )

            logger.write([
                global_step,
                reply["loss"],
                payload_mb,
                capture_ms,
                send_recv_ms,
                state_load_ms,
                total_ms,
            ])

            global_step += 1

    test_loss, test_acc = evaluate(model, test_loader,device=device)

    if grad_save_path is not None:
        torch.save(model.state_dict(), "split_final.pt")
    else:
        torch.save(model.state_dict(), "split_final.pt")

    print(
        f"[Eval] "
        f"test_loss={test_loss:.6f} "
        f"test_acc={test_acc * 100:.2f}%"
    )

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