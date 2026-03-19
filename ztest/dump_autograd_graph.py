# dump_autograd_graph.py
# Usage:
#   python dump_autograd_graph.py
#   python dump_autograd_graph.py --cuda
#   python dump_autograd_graph.py --cuda --device cuda:1

import argparse
import torch
import torch.nn.functional as F

from torchvision.models import resnet18

class LeNetLike(torch.nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, 6, kernel_size=5, stride=1, padding=0)
        self.pool  = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1   = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2   = torch.nn.Linear(120, 84)
        self.fc3   = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        x, idx1 = self.pool(F.relu(self.conv1(x)))
        x, idx2 = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def walk(fn, depth=0, seen=None, max_depth=50):
    if fn is None:
        return
    if seen is None:
        seen = set()
    if fn in seen:
        return
    seen.add(fn)

    print("  " * depth + f"- {type(fn)}")
    if depth >= max_depth:
        print("  " * (depth + 1) + "(max_depth reached)")
        return

    for nxt, _ in fn.next_functions:
        if nxt is not None:
            walk(nxt, depth + 1, seen, max_depth=max_depth)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", help="use CUDA if available")
    ap.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu")
    args = ap.parse_args()

    # device 결정
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    # resnet 
    # model = LeNetLike(in_ch=1, num_classes=10).to(device)

    model = resnet18(weights=None).to(device)
    model.train()

    # lenet
    # x = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)

    x = torch.randn(1,3,224,224,device =device, requires_grad=True)
    y = model(x)
    loss = y.sum()

    print(f"=== device: {device} ===")
    print("=== model ===")
    print(model)
    print("\n=== loss.grad_fn ===")
    print(type(loss.grad_fn), "\n")

    print("=== autograd graph (backward nodes) ===")
    walk(loss.grad_fn)

if __name__ == "__main__":
    main()