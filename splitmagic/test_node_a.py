import torch

from splitmagic.models import LeNet
from splitmagic.models import LongCNN
from splitmagic.data import make_cifar10_loaders
from splitmagic.node_a_runtime import run_node_a


def main():
    torch.manual_seed(0)

    train_loader, test_loader = make_cifar10_loaders(
        root="./data",
        batch_size=16,
        test_batch_size=128,
        shuffle=True,
        num_workers=2,
    )

    # model = LeNet()
    model = LongCNN()

    run_node_a(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        endpoint="tcp://127.0.0.1:5555",
        csv_path="node_a_timing.csv",
        num_epochs=10,
        max_steps=10,
        policy="auto_recompute",
        # optional_keys=[
        #     "maxpool2d:0:indices",
        #     "maxpool2d:1:indices",
        # ]
    )


if __name__ == "__main__":
    main()
