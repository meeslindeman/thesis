"""
Data utilities for IDMM experiments (small datasets).
All datasets download automatically via torchvision.

Training splits follow the paper (Table 1):
  flowers102 — train+val (2040),  test (6149)
  pets       — trainval  (3680),  test (3669)
  dtd        — train+val (3760),  test (1880)
  aircrafts  — trainval  (6667),  test (3333)
"""

import torch
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

DATASET_REGISTRY = {
    "flowers102": {"image_size": 224, "num_classes": 102},
    "pets":       {"image_size": 224, "num_classes": 37},
    "dtd":        {"image_size": 224, "num_classes": 47},
    "aircrafts":  {"image_size": 224, "num_classes": 100},
}


def _build_transforms(size: int, train: bool):
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _load_splits(name: str, root: Path, train_tf, test_tf):
    """
    Returns (train_full, val_src, test).
    val_src mirrors train_full with test_tf (no augmentation), used for validation splitting.
    Training sets match the paper's split sizes by combining train+val where applicable.
    """
    if name == "flowers102":
        from torchvision.datasets import Flowers102
        # train (1020) + val (1020) = 2040 as in the paper
        train_full = ConcatDataset([
            Flowers102(root, split="train", transform=train_tf, download=True),
            Flowers102(root, split="val",   transform=train_tf, download=False),
        ])
        val_src = ConcatDataset([
            Flowers102(root, split="train", transform=test_tf, download=False),
            Flowers102(root, split="val",   transform=test_tf, download=False),
        ])
        test = Flowers102(root, split="test", transform=test_tf, download=False)

    elif name == "pets":
        from torchvision.datasets import OxfordIIITPet
        # trainval = 3680 as in the paper
        train_full = OxfordIIITPet(root, split="trainval", transform=train_tf, download=True)
        val_src    = OxfordIIITPet(root, split="trainval", transform=test_tf,  download=False)
        test       = OxfordIIITPet(root, split="test",     transform=test_tf,  download=False)

    elif name == "dtd":
        from torchvision.datasets import DTD
        # train (1880) + val (1880) = 3760 as in the paper
        train_full = ConcatDataset([
            DTD(root, split="train", transform=train_tf, download=True),
            DTD(root, split="val",   transform=train_tf, download=False),
        ])
        val_src = ConcatDataset([
            DTD(root, split="train", transform=test_tf, download=False),
            DTD(root, split="val",   transform=test_tf, download=False),
        ])
        test = DTD(root, split="test", transform=test_tf, download=False)

    elif name == "aircrafts":
        from torchvision.datasets import FGVCAircraft
        # trainval = 6667 as in the paper
        train_full = FGVCAircraft(root, split="trainval", transform=train_tf, download=True)
        val_src    = FGVCAircraft(root, split="trainval", transform=test_tf,  download=False)
        test       = FGVCAircraft(root, split="test",     transform=test_tf,  download=False)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    return train_full, val_src, test


def get_dataset(name: str, data_dir: str = "data/", validation_size: int = 0, train_transform=None):
    """
    Returns ({"train": dataset, "validation": dataset, "test": dataset}, num_classes).
    Pass train_transform to override the default (e.g. MultiCropTransform for pretraining).
    """
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    meta = DATASET_REGISTRY[name]
    root = Path(data_dir)
    train_tf = train_transform if train_transform is not None else _build_transforms(meta["image_size"], train=True)
    test_tf  = _build_transforms(meta["image_size"], train=False)

    train_full, val_src, test = _load_splits(name, root, train_tf, test_tf)
    num_classes = meta["num_classes"]

    if validation_size > 0:
        indices = torch.randperm(
            len(train_full), generator=torch.Generator().manual_seed(42)
        ).tolist()
        train = torch.utils.data.Subset(train_full, indices[validation_size:])
        val   = torch.utils.data.Subset(val_src,    indices[:validation_size])
    else:
        train = train_full
        val   = torch.utils.data.Subset(val_src, [])

    return {"train": train, "validation": val, "test": test}, num_classes


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False):
    return {
        "train":      DataLoader(dataset["train"],      batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=8, pin_memory=True),
        "validation": DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True),
        "test":       DataLoader(dataset["test"],       batch_size=256,        shuffle=False,   num_workers=4, pin_memory=True),
    }
