import torch

from splitmagic.models import LeNet
from splitmagic.models import LongCNN
from splitmagic.node_b_runtime import run_node_b


def main():
    torch.manual_seed(0)

    # model = LeNet()
    model = LongCNN()

    run_node_b(
        model=model,
        endpoint="tcp://*:5555",
        csv_path="node_b_timing.csv",
        lr=0.1,
    )

if __name__ == "__main__":
    main()