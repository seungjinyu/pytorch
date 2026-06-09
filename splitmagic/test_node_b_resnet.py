import os
import random
import numpy as np
import torch

from splitmagic.node_b_runtime import run_node_b
from splitmagic.models import make_resnet18_cifar10


SEED = 1234
INIT = "./resnet18_init_state.pt"


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seed_all()

    model = make_resnet18_cifar10()

    if not os.path.exists(INIT):
        torch.save(model.state_dict(), INIT)
        print(f"[Node B] saved init state: {INIT}")

    model.load_state_dict(torch.load(INIT, map_location="cpu"))
    model.train()

    run_node_b(
        model=model,
        endpoint="tcp://*:5555",
        csv_path="node_b_vgg11bn_grad.csv",
        lr=0.1,
    )


if __name__ == "__main__":
    main()