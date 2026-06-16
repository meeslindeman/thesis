"""
Data utilities for IDMM experiments (small datasets).
Most datasets download automatically via torchvision; CUB-200-2011 must be
placed manually at data/CUB_200_2011/ (download from https://www.vision.caltech.edu/datasets/cub_200_2011/).

Training splits follow the paper (Table 1):
  flowers102 — train+val (2040),  test (6149)
  dtd        — train+val (3760),  test (1880)
  aircrafts  — trainval  (6667),  test (3333)
  cub200     — train     (5994),  test (5794)
"""

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

DATASET_REGISTRY = {
    "flowers102": {"image_size": 224, "num_classes": 102},
    "dtd":        {"image_size": 224, "num_classes": 47},
    "aircrafts":  {"image_size": 224, "num_classes": 100},
    "cub200":     {"image_size": 224, "num_classes": 200},
}


class CUB200(Dataset):
    """CUB-200-2011 fine-grained bird classification dataset.
    Expects the standard extracted layout at root/CUB_200_2011/.
    Labels are 0-indexed (original files use 1-indexed class ids).
    """
    def __init__(self, root: Path, train: bool = True, transform=None):
        self.root = Path(root) / "CUB_200_2011"
        self.transform = transform

        images = {}
        with open(self.root / "images.txt") as f:
            for line in f:
                img_id, fname = line.strip().split()
                images[int(img_id)] = fname

        splits = {}
        with open(self.root / "train_test_split.txt") as f:
            for line in f:
                img_id, is_train = line.strip().split()
                splits[int(img_id)] = int(is_train)

        labels = {}
        with open(self.root / "image_class_labels.txt") as f:
            for line in f:
                img_id, class_id = line.strip().split()
                labels[int(img_id)] = int(class_id) - 1

        self.samples = [
            (self.root / "images" / images[img_id], labels[img_id])
            for img_id, is_train in sorted(splits.items())
            if bool(is_train) == train
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _build_transforms(size: int, train: bool):
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandAugment(),
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

    elif name == "cub200":
        # train=5994, test=5794; no official val split
        train_full = CUB200(root, train=True,  transform=train_tf)
        val_src    = CUB200(root, train=True,  transform=test_tf)
        test       = CUB200(root, train=False, transform=test_tf)

    else:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    return train_full, val_src, test


def get_dataset(name: str, data_dir: str = "data/", validation_size: int = 0, train_transform=None, image_size: int = None):
    """
    Returns ({"train": dataset, "validation": dataset, "test": dataset}, num_classes).
    Pass train_transform to override the default (e.g. MultiCropTransform for pretraining).
    Pass image_size to override the registry default (e.g. 448 for the high-res fine-tune stage).
    """
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    meta = DATASET_REGISTRY[name]
    root = Path(data_dir)
    size = image_size if image_size is not None else meta["image_size"]
    train_tf = train_transform if train_transform is not None else _build_transforms(size, train=True)
    test_tf  = _build_transforms(size, train=False)

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
        "train":      DataLoader(dataset["train"],      batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=8, pin_memory=True, persistent_workers=True),
        "validation": DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True, persistent_workers=True),
        "test":       DataLoader(dataset["test"],       batch_size=256,        shuffle=False,   num_workers=4, pin_memory=True, persistent_workers=True),
    }
