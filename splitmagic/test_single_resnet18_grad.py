import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18


SEED = 1234
BS = 32
OUT = "/tmp/baseline_grads.pt"


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


def main():
    seed_all()

    model = make_model()
    model.train()
    model.zero_grad(set_to_none=True)

    x, y = make_batch()

    loss_fn = nn.CrossEntropyLoss()
    out = model(x)
    loss = loss_fn(out, y)

    print(f"[BASELINE] loss={loss.item():.10f}")

    loss.backward()

    grads = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().cpu().clone()
            print(
                f"[BASELINE][GRAD] {name:40s} "
                f"mean={p.grad.mean().item():.10f} "
                f"absmax={p.grad.abs().max().item():.10f}"
            )

    torch.save(grads, OUT)
    print(f"[BASELINE] saved {OUT}")


if __name__ == "__main__":
    main()