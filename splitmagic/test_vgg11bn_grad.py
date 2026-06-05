import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


SEED = 1234
BS = 32
OUT = "./vgg_baseline_grads.pt"


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model():
    model = torchvision.models.vgg11_bn(num_classes=10)

    model.avgpool = nn.Identity()

    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 10),
    )

    return model


def make_batch():
    ds = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=T.ToTensor(),
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
    # JIN overwrite 환경변수 제거
    for k in [
        "JIN_ROLE",
        "JIN_PAYLOAD_PATH",
        "JIN_STEP",
        "JIN_ALIAS_PATH",
    ]:
        os.environ.pop(k, None)

    seed_all()

    device = torch.device("cpu")

    model = make_model().to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    x, y = make_batch()
    x = x.to(device)
    y = y.to(device)

    loss_fn = nn.CrossEntropyLoss()

    out = model(x)
    loss = loss_fn(out, y)

    print(f"[LOCAL][LOSS] {loss.item():.10f}")

    loss.backward()

    grads = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        grads[name] = p.grad.detach().cpu().clone()

        print(
            f"[LOCAL][GRAD] {name:40s} "
            f"mean={p.grad.mean().item():.10f} "
            f"absmax={p.grad.abs().max().item():.10f}"
        )

    torch.save(grads, OUT)
    print(f"[LOCAL] saved {OUT}")


if __name__ == "__main__":
    main()