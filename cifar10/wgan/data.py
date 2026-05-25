from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

MASK_SIZE = 16
IMAGE_SIZE = 32


def apply_masks(images, idy, idx):
    b, _, h, w = images.shape
    mask = torch.zeros(b, 1, h, w)
    for k in range(b):
        mask[k, :, idy[k]:idy[k] + MASK_SIZE, idx[k]:idx[k] + MASK_SIZE] = 1

    masked = images * (1 - mask)
    y = images
    x = torch.cat([masked, mask], dim=1)
    return x, y


def make_masked_batch(batch):
    images = torch.stack([image for image, _ in batch])
    b, _, h, w = images.shape

    idy = torch.randint(0, h - MASK_SIZE + 1, (b,))
    idx = torch.randint(0, w - MASK_SIZE + 1, (b,))

    return apply_masks(images, idy, idx)


def make_fixed_masked_batch(batch):
    images = torch.stack([image for image, _, _ in batch])
    idy = torch.tensor([p_idy for _, p_idy, _ in batch])
    idx = torch.tensor([p_idx for _, _, p_idx in batch])

    return apply_masks(images, idy, idx)


class FixedMaskDataset(Dataset):
    def __init__(self, base, idy, idx):
        self.base = base
        self.idy = idy
        self.idx = idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        image, _ = self.base[k]
        return image, int(self.idy[k]), int(self.idx[k])


def get_dataloaders(
    root=DATA_ROOT,
    train_size=49_000,
    val_size=1_000,
    test_size=10_000,
    batch_size=128,
    seed=0,
):
    generator = torch.Generator().manual_seed(seed)
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_full = CIFAR10(root=root, train=True, download=True, transform=transform)
    test_full = CIFAR10(root=root, train=False, download=True, transform=transform)

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

    val_idy = torch.randint(0, IMAGE_SIZE - MASK_SIZE + 1, (len(val),), generator=generator)
    val_idx = torch.randint(0, IMAGE_SIZE - MASK_SIZE + 1, (len(val),), generator=generator)
    test_idy = torch.randint(0, IMAGE_SIZE - MASK_SIZE + 1, (len(test),), generator=generator)
    test_idx = torch.randint(0, IMAGE_SIZE - MASK_SIZE + 1, (len(test),), generator=generator)

    val = FixedMaskDataset(val, val_idy, val_idx)
    test = FixedMaskDataset(test, test_idy, test_idx)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_masked_batch,
    )
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=make_fixed_masked_batch)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=make_fixed_masked_batch)

    return train_loader, val_loader, test_loader
