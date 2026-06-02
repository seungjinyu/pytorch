import torch

from splitmagic.node_b_runtime import run_node_b
from splitmagic.models import make_resnet18_cifar10


def main():
    torch.manual_seed(0)

    model = make_resnet18_cifar10()

    state = torch.load("resnet18_init_state.pt")

    model.load_state_dict(state)
    model.eval()

    run_node_b(
        model=model,
    )


if __name__ == "__main__":
    main()