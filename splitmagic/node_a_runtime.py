import time
import torch
import torch.nn.functional as F

from splitmagic import SplitRuntime, ZMQClient
from splitmagic.utils.timing import CSVLogger

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
    key_mode="shape",
):

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

            policy_meta = payload.meta.get("tensor_policy", {})

            t_capture1 = time.perf_counter()

            extra = {
                "tensor_policy": policy_meta,
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