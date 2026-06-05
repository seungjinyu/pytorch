import os
import random
import numpy as np
import torch

from splitmagic.node_a_runtime import run_node_a
from splitmagic.models import make_vgg11_bn_cifar10
from splitmagic.data import make_cifar10_loaders


SEED = 1234
INIT = "./vgg11bn_init_state.pt"
SPLIT_GRADS = "./vgg_split_grads.pt"


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seed_all()

    train_loader, test_loader = make_cifar10_loaders(
        batch_size=32,
        test_batch_size=128,
        shuffle=False,
    )

    model = make_vgg11_bn_cifar10()

    if not os.path.exists(INIT):
        torch.save(model.state_dict(), INIT)
        print(f"[Node A] saved init state: {INIT}")

    model.load_state_dict(torch.load(INIT, map_location="cpu"))
    model.train()

    run_node_a(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        endpoint="tcp://127.0.0.1:5555",
        csv_path="node_a_vgg11bn_grad.csv",
        num_epochs=1,
        max_steps=1000,
        policy="full",
        key_mode="graph",
        grad_save_path=SPLIT_GRADS,
    )


if __name__ == "__main__":
    main()