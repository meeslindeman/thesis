import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

DATASET_REGISTRY = {
    "cifar10": {
        "cls": CIFAR10,
        "num_classes": 10,
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std":  (0.24703223, 0.24348513, 0.26158784),
    },
    "cifar100": {
        "cls": CIFAR100,
        "num_classes": 100,
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std":  (0.26733429, 0.25643846, 0.27615047),
    },
}


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, return_numpy=False):
    collate_fn = numpy_collate_fn if return_numpy else None
    return {
        "train": DataLoader(dataset["train"], batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn),
        "validation": DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn),
        "test": DataLoader(dataset["test"], batch_size=512, shuffle=False, drop_last=False, collate_fn=collate_fn)
    }


def numpy_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0).numpy()
    labels = np.array([b[1] for b in batch], dtype=np.int32)
    return imgs, labels


def _build_transforms(meta, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ])


def get_dataset(name, data_dir="data/", validation_size=5000):
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    meta = DATASET_REGISTRY[name]
    cls = meta["cls"]

    train_transform = _build_transforms(meta, train=True)
    test_transform = _build_transforms(meta, train=False)

    train_full = cls(root=data_dir, train=True, download=True, transform=train_transform)
    val_full = cls(root=data_dir, train=True, download=True, transform=test_transform)
    test = cls(root=data_dir, train=False, download=True, transform=test_transform)

    if not 0 <= validation_size <= len(train_full):
        raise ValueError(f"validation_size must be in [0, {len(train_full)}], got {validation_size}.")

    indices = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[validation_size:]
    val_indices = indices[:validation_size]

    train = torch.utils.data.Subset(train_full, train_indices)
    val = torch.utils.data.Subset(val_full, val_indices)

    return {"train": train, "validation": val, "test": test}, meta["num_classes"]