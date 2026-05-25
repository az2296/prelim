from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def make_masked_batch(batch):
    images = torch.stack([image for image, _ in batch])

    y = images[:, :, 7:21, 7:21]

    masked = images.clone()
    masked[:, :, 7:21, 7:21] = 0

    mask = torch.zeros_like(images)
    mask[:, :, 7:21, 7:21] = 1

    x = torch.cat([masked, mask], dim=1)
    return x, y

def get_dataloaders(
    root=DATA_ROOT,
    train_size=20_000,
    val_size=1_000,
    test_size=10_000,
    batch_size=128,
    seed=0,
):
    generator = torch.Generator().manual_seed(seed)
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    train_full = MNIST(root=root, train=True, download=True, transform=transform)
    test_full = MNIST(root=root, train=False, download=True, transform=transform)

    train, val, _ = random_split(
        train_full,
        [train_size, val_size, len(train_full) - train_size - val_size],
        generator=generator,
    )
    test, _ = random_split(
        test_full,
        [test_size, len(test_full) - test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_masked_batch,
    )
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=make_masked_batch)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=make_masked_batch)

    return train_loader, val_loader, test_loader
