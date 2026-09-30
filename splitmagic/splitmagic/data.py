from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from torchvision import  transforms,datasets
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

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

def make_imagenet_loaders(
    data_root,
    batch_size=4,
    test_batch_size=4,
    shuffle=False,
    num_workers=0,
):
    data_root = Path(data_root)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_dataset = datasets.ImageFolder(
        root=data_root / "train",
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        root=data_root / "val",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader

def make_tinystories_test_loader(
    batch_size=1,
    sequence_length=9,
    vocab_size=50257,
):
    # 우선 framework 검증용 deterministic token IDs
    input_ids = torch.arange(
        100,
        100 + batch_size * sequence_length,
        dtype=torch.long,
    ).reshape(
        batch_size,
        sequence_length,
    )

    input_ids = input_ids % vocab_size

    # causal LM training용 label
    labels = input_ids.clone()

    dataset = TensorDataset(
        input_ids,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return loader