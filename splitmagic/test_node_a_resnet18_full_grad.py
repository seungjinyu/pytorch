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


SEED = 1234
BS = 32
HOST = "127.0.0.1"
PORT = 5555

PAYLOAD_PATH = "/tmp/jin_payload_send.bin"


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


def send_bytes(sock, data):
    sock.sendall(struct.pack("<Q", len(data)))
    sock.sendall(data)


def main():
    seed_all()

    model = make_model()
    runtime = SplitRuntime(model, role="A")

    x, y = make_batch()
    loss_fn = nn.CrossEntropyLoss()

    payload = runtime.capture_jin(
        x=x,
        y=y,
        loss_fn=loss_fn,
        policy="full",
        key_mode="graph",
    )

    payload.save_jin1(PAYLOAD_PATH)

    with open(PAYLOAD_PATH, "rb") as f:
        payload_bytes = f.read()

    tensor_policy = payload.meta["tensor_policy"]

    meta = {
        "model_output": payload.tensors["model.output"],
        "tensor_policy": tensor_policy,
    }

    meta_bytes = pickle.dumps(meta)

    print(f"[Node A] send payload bytes={len(payload_bytes)}")
    print(f"[Node A] tensor_policy keys={len(tensor_policy.get('all_keys', []))}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    send_bytes(sock, payload_bytes)
    send_bytes(sock, meta_bytes)

    reply = sock.recv(1024)
    print("[Node A] reply:", reply)

    sock.close()


if __name__ == "__main__":
    main()