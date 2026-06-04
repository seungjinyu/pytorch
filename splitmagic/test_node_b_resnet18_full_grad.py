import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import socket
import struct
import pickle
from torchvision.models import resnet18

from splitmagic.runtime import SplitRuntime
from splitmagic.payload import Payload


SEED = 1234
BS = 32
HOST = "127.0.0.1"
PORT = 5555

PAYLOAD_PATH = "/tmp/jin_payload_recv.bin"
OUT = "/tmp/split_grads.pt"


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model():
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def make_batch():
    transform = T.Compose([
        T.ToTensor(),
    ])

    ds = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BS,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    return next(iter(loader))


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed")
        buf += chunk
    return buf


def recv_bytes(conn):
    raw_len = recv_exact(conn, 8)
    n = struct.unpack("<Q", raw_len)[0]
    return recv_exact(conn, n)


def main():
    seed_all()

    model = make_model()
    runtime = SplitRuntime(model, role="B")

    x_dummy, y = make_batch()

    loss_fn = nn.CrossEntropyLoss()

    print("[Node B] listening")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    conn, addr = server.accept()
    print(f"[Node B] connected from {addr}")

    try:
        payload_bytes = recv_bytes(conn)
        print(f"[RECV_PAYLOAD_BYTES] {len(payload_bytes)}")

        with open(PAYLOAD_PATH, "wb") as f:
            f.write(payload_bytes)

        meta_bytes = recv_bytes(conn)
        meta = pickle.loads(meta_bytes)

        payload = Payload()
        payload.add_tensor("model.output", meta["model_output"])
        payload.meta["tensor_policy"] = meta["tensor_policy"]

        print("[RECV_KEYS]", list(payload.tensors.keys()))
        print("[Node B][PAYLOAD_KEYS]", list(payload.tensors.keys()))

        loss = runtime.backward_jin(
            x_dummy=x_dummy,
            y=y,
            payload=payload,
            loss_fn=loss_fn,
            payload_path=PAYLOAD_PATH,
            tensor_policy=meta["tensor_policy"],
        )

        print(f"[Node B] loss={loss.item():.10f}")

        grads = {}

        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().cpu().clone()

        torch.save(grads, OUT)
        print(f"[Node B] saved {OUT}")

        conn.sendall(b"OK")

    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()