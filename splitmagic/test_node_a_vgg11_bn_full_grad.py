import os
import torch

from splitmagic.node_a_runtime import run_node_a
from splitmagic.models import make_vgg11_bn_cifar10
from splitmagic.data import make_cifar10_loaders


def main():
    torch.manual_seed(0)

    train_loader, test_loader = make_cifar10_loaders(batch_size=32, shuffle=False)

    model = make_vgg11_bn_cifar10()

    if not os.path.exists("vgg11_bn_init_state.pt"):
        torch.save(model.state_dict(), "vgg11_bn_init_state.pt")

    model.load_state_dict(torch.load("vgg11_bn_init_state.pt"))

    run_node_a(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,
        max_steps=1,
        policy="full",
        grad_save_path="vgg11_bn_grads_full.pt",
    )


if __name__ == "__main__":
    main()
