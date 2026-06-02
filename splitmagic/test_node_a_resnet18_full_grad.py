import os
import torch

from splitmagic.node_a_runtime import run_node_a
from splitmagic.models import make_resnet18_cifar10
from splitmagic.data import make_cifar10_loaders


def main():
    torch.manual_seed(0)

    train_loader, test_loader = make_cifar10_loaders(
        batch_size=32,
        test_batch_size=128,
        shuffle=False,
    )

    model = make_resnet18_cifar10()
    

    if not os.path.exists("resnet18_init_state.pt"):
        torch.save(model.state_dict(), "resnet18_init_state.pt")

    model.load_state_dict(torch.load("resnet18_init_state.pt"))
    model.eval()
    run_node_a(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=1,
        max_steps=1,
        policy="full",
        # key_mode="autograd_order",
        grad_save_path="resnet18_grads_full.pt",
    )


if __name__ == "__main__":
    main()