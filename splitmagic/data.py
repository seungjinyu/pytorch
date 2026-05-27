from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

def make_cifar10_loaders(
    root="./data",
    batch_size=32,
    test_batch_size=128,
    shuffle=True,
    num_workers=2,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader