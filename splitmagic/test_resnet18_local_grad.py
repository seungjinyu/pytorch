import os
import torch
import torch.nn.functional as F

from splitmagic.models import make_resnet18_cifar10
from splitmagic.data import make_cifar10_loaders


def clone_grads(model):
    return {
        name: p.grad.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }

def dump_backward_graph(loss):
    visited = set()

    def walk(fn, depth=0):
        if fn is None:
            return
        if id(fn) in visited:
            return

        visited.add(id(fn))

        print("  " * depth + type(fn).__name__)

        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)

    walk(loss.grad_fn)


def main():
    torch.manual_seed(0)

    train_loader, _ = make_cifar10_loaders(
        batch_size=32,
        shuffle=False,
    )

    x, y = next(iter(train_loader))

    model = make_resnet18_cifar10()

    if not os.path.exists("resnet18_init_state.pt"):
        torch.save(model.state_dict(), "resnet18_init_state.pt")

    model.load_state_dict(torch.load("resnet18_init_state.pt"))

    model.train()
    # model.eval()
    model.zero_grad(set_to_none=True)

    out = model(x)
    print("[DEBUG] logits_sum", out.detach().sum().item())
    print("[DEBUG] logits_mean", out.detach().mean().item())
    print("[DEBUG] label_sum", y.sum().item())
    loss = F.cross_entropy(out, y)

    dump_backward_graph(loss)

    print(f"[LOCAL] loss={loss.item():.6f}")

    loss.backward()

    torch.save(clone_grads(model), "resnet18_grads_local.pt")

    print("[LOCAL] saved grads to resnet18_grads_local.pt")


if __name__ == "__main__":
    main()