import os
import torch
import torch.nn.functional as F

from splitmagic.models import make_vgg11_bn_cifar10
from splitmagic.data import make_cifar10_loaders


def clone_grads(model):
    return {
        name: p.grad.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def main():
    torch.manual_seed(0)

    train_loader, _ = make_cifar10_loaders(batch_size=32, shuffle=False)
    x, y = next(iter(train_loader))

    model = make_vgg11_bn_cifar10()

    if not os.path.exists("vgg11_bn_init_state.pt"):
        torch.save(model.state_dict(), "vgg11_bn_init_state.pt")

    model.load_state_dict(torch.load("vgg11_bn_init_state.pt"))

    model.train()
    model.zero_grad(set_to_none=True)

    out = model(x)
    loss = F.cross_entropy(out, y)

    print(f"[LOCAL] loss={loss.item():.6f}")

    loss.backward()

    torch.save(clone_grads(model), "vgg11_bn_grads_local.pt")

    print("[LOCAL] saved grads to vgg11_bn_grads_local.pt")


if __name__ == "__main__":
    main()