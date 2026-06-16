import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18


def make_resnet18_cifar10():
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()

    return loss_sum / total, 100.0 * correct / total


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "model_state",
            "net",
        ]:
            if key in ckpt:
                print(f"[LOAD] using ckpt['{key}']")
                return ckpt[key]

    print("[LOAD] using checkpoint directly as state_dict")
    return ckpt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "split_final.pt"

    model = make_resnet18_cifar10()

    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("[CKPT TYPE]", type(ckpt))
    if isinstance(ckpt, dict):
        print("[CKPT KEYS]", ckpt.keys())

    state = extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[LOAD] missing keys:", missing)
    print("[LOAD] unexpected keys:", unexpected)

    if missing or unexpected:
        raise RuntimeError("Checkpoint does not match model structure")

    model.to(device)

    # 학습 때 ToTensor만 썼으므로 평가도 동일하게
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_set = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=2,
    )

    test_loss, test_acc = evaluate(model, test_loader, device)

    print(f"[Eval] test_loss={test_loss:.6f} test_acc={test_acc:.2f}%")


if __name__ == "__main__":
    main()