import time
import torch
import torch.nn.functional as F
import os 

from splitmagic import SplitRuntime, ZMQClient
from splitmagic.utils.timing import CSVLogger
from splitmagic.runtime import read_dryrun_plan, keys_from_dryrun_plan


def clone_state_dict(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

def load_dryrun_tensors_into_payload(payload, tensor_dir):

    new_tensors = {}

    # print("[BEFORE_DRYRUN_LOAD_KEYS]",len(payload.tensors),sorted(payload.tensors.keys())[:20])

    if "model.output" in payload.tensors:
        new_tensors["model.output"] = payload.tensors["model.output"]

    for name in os.listdir(tensor_dir):
        if not name.endswith(".pt"):
            continue

        key = name[:-3]
        path = os.path.join(tensor_dir, name)

        try:
            obj = torch.jit.load(path, map_location="cpu")
            t = obj.tensor
        except Exception as e:
            # print("[DRYRUN_LOAD_FAIL]", key, type(e), e)
            continue

        # print("[DRYRUN_LOAD]", key, type(t), tuple(t.shape))

        new_tensors[key] = t.detach().cpu().contiguous()

    payload.tensors = new_tensors
    # print("[AFTER_DRYRUN_LOAD_KEYS]", len(payload.tensors), sorted(payload.    tensors.keys())[:30])
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
    dryrun_plan = False,
):
    
    os.environ["JIN_ROLE"] = "A"    
    os.environ.pop("JIN_DRYRUN", None)
    dryrun_done = False

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

    for epoch in range(num_epochs):

        for _, (x, y) in enumerate(train_loader):

            if global_step >= max_steps:
                break

            if dryrun_plan and not dryrun_done:

                # state_before_dryrun = clone_state_dict(model)

                dryrun_path = "/tmp/jin_dryrun_plan.tsv"

                if os.path.exists(dryrun_path):
                    os.remove(dryrun_path)

                import shutil
                os.environ["JIN_DRYRUN"] = "1"
                os.environ["JIN_DRYRUN_TENSOR_DIR"] = "/tmp/jin_dryrun_tensors"

                shutil.rmtree("/tmp/jin_dryrun_tensors", ignore_errors=True)
                os.makedirs("/tmp/jin_dryrun_tensors", exist_ok=True)
                os.environ["JIN_DRYRUN_PATH"] = dryrun_path

                model.zero_grad(set_to_none=True)

                # x_dry = torch.randn_like(x)

                # y_dry = torch.randint(
                #     0,
                #     10,
                #     (x.size(0),),
                #     device=x.device,
                # )

                # out = model(x_dry)
                # loss = F.cross_entropy(out, y_dry)
                # loss.backward()

                out = model(x)
                loss = F.cross_entropy(out,y)
                loss.backward()

                os.environ.pop("JIN_DRYRUN", None)
                os.environ.pop("JIN_DRYRUN_PATH",None)
                os.environ.pop("JIN_DRYRUN_TENSOR_DIR", None)

                # model.load_state_dict(state_before_dryrun)
                model.zero_grad(set_to_none=True)

                dryrun_done = True

            t0 = time.perf_counter()

            t_capture0 = time.perf_counter()

            payload = runtime_a.capture_jin(
                x=x,
                y=y,
                loss_fn=F.cross_entropy,
                policy=policy, 
                optional_keys=optional_keys, 
                key_mode=key_mode, 
            )

            if dryrun_plan:
                plan = read_dryrun_plan("/tmp/jin_dryrun_plan.tsv")
                payload.meta["dryrun_backward_plan"] = plan
                # 일단 remap 끄기
                # payload = remap_payload_to_dryrun_idx(payload, plan)
                payload = load_dryrun_tensors_into_payload(
                    payload,
                    "/tmp/jin_dryrun_tensors"
                )

                needed_keys = keys_from_dryrun_plan(plan)
                print("[NEEDED_KEYS]", len(needed_keys), sorted(needed_keys)[:50])
                always_keep = {"model.output"}

                before_count = len(payload.tensors)

                print("[PAYLOAD_KEYS_BEFORE_FILTER]",
                    len(payload.tensors),
                    sorted(payload.tensors.keys())[:20])

                print("[NEEDED_KEYS]",
                    len(needed_keys),
                    sorted(list(needed_keys))[:20])

                payload.tensors = {
                    k: v for k, v in payload.tensors.items()
                    if k in needed_keys or k in always_keep
                }

                print("[PAYLOAD_KEYS_AFTER_FILTER]",
                    len(payload.tensors),
                    sorted(payload.tensors.keys())[:20])

                payload.meta["tensor_policy"] = {
                    "policy": "dryrun_plan_keys",
                    "before_count": before_count,
                    "num_payload_tensors": len(payload.tensors),
                    "included_keys": sorted(payload.tensors.keys()),
                }

            policy_meta = payload.meta.get("tensor_policy", {})

            t_capture1 = time.perf_counter()

            extra = {
                "tensor_policy": policy_meta,
                "dryrun_backward_plan":payload.meta.get("dryrun_backward_plan",[]),
            }

            if global_step == 0:
                extra["state_dict"]= clone_state_dict(model)

            t_send0 = time.perf_counter()

            reply = client.send_payload(
                payload=payload,
                y=y,
                batch_size=x.size(0),
                extra=extra,
            )

            if grad_save_path is not None and "grads" in reply:
                torch.save(reply["grads"], grad_save_path)
                print(f"[Node A] saved grads to {grad_save_path}")
                return


            t_send1 = time.perf_counter()

            if reply["status"] != "ok":
                print("[Node A] bad reply:", reply)
                break
            
            t_load0 = time.perf_counter()

            if "grads" in reply:
                print("[Node A][GRADS] received:", sorted(reply["grads"].keys()))

            model.load_state_dict(reply["updated_state_dict"])

            t_load1 = time.perf_counter()

            total_ms = (time.perf_counter() - t0) * 1000

            capture_ms = (t_capture1 - t_capture0) * 1000
            send_recv_ms = (t_send1 - t_send0) * 1000
            state_load_ms = (t_load1 - t_load0) * 1000

            payload_mb = reply["bytes"] / 1024 / 1024

            print(
                f"[Node A] epoch={epoch} "
                f"step={global_step} "
                f"loss={reply['loss']:.6f} "
                f"payload_mb={payload_mb:.3f} "
                f"capture_ms={capture_ms:.2f} "
                f"send_recv_ms={send_recv_ms:.2f} "
                f"state_load_ms={state_load_ms:.2f} "
                f"total_ms={total_ms:.2f}"
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

    test_loss, test_acc = evaluate(model, test_loader)
    
    torch.save(model.state_dict(), "vgg11_bn_split_final.pt")

    print(
        f"[Eval] "
        f"test_loss={test_loss:.6f} "
        f"test_acc={test_acc * 100:.2f}%"
    )

    print("[Node A] done")

@torch.no_grad()
def evaluate_accuracy(model, test_loader, device="cpu"):
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

    acc = 100.0 * correct / total
    avg_loss = total_loss / total

    model.train()
    return avg_loss, acc

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

    return total_loss / total, correct / total
