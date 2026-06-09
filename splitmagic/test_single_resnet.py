import os
import random
import numpy as np
import torch
import torch.nn as nn

from splitmagic.models import make_resnet18_cifar10
from splitmagic.data import make_cifar10_loaders


SEED = 1234
INIT = "./resnet18_init_state.pt"
OUT = "./resnet18_baseline_grads.pt"


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    for k in [
        "JIN_ROLE",
        "JIN_PAYLOAD_PATH",
        "JIN_STEP",
        "JIN_ALIAS_PATH",
    ]:
        os.environ.pop(k, None)

    seed_all()

    train_loader, _ = make_cifar10_loaders(
        batch_size=32,
        test_batch_size=128,
        shuffle=False,
    )

    x, y = next(iter(train_loader))

    model = make_resnet18_cifar10()

    if not os.path.exists(INIT):
        torch.save(model.state_dict(), INIT)
        print(f"[LOCAL] saved init state: {INIT}")

    model.load_state_dict(torch.load(INIT, map_location="cpu"))
    model.train()
    model.zero_grad(set_to_none=True)

    loss_fn = nn.CrossEntropyLoss()

    out = model(x)
    loss = loss_fn(out, y)

    print(f"[LOCAL][LOSS] {loss.item():.10f}")

    loss.backward()

    grads = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().cpu().clone()

    torch.save(grads, OUT)
    print(f"[LOCAL] saved {OUT}")


if __name__ == "__main__":
    main()