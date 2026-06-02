import urllib.request
import zipfile
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

_TINY_IN_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = Path(root)
        self.data_dir = self.root / "tiny-imagenet-200"
        self.transform = transform

        if download:
            self._download()
        if not self.data_dir.exists():
            raise RuntimeError("Tiny ImageNet not found. Pass download=True.")

        if train:
            folder = ImageFolder(self.data_dir / "train")
            self.samples = folder.samples
        else:
            wnids = sorted((self.data_dir / "wnids.txt").read_text().strip().split("\n"))
            class_to_idx = {w: i for i, w in enumerate(wnids)}
            img_dir = self.data_dir / "val" / "images"
            self.samples = [
                (str(img_dir / parts[0]), class_to_idx[parts[1]])
                for line in (self.data_dir / "val" / "val_annotations.txt").read_text().strip().split("\n")
                for parts in [line.split("\t")]
            ]

    def _download(self):
        if self.data_dir.exists():
            return
        zip_path = self.root / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            print("Downloading Tiny ImageNet...")
            urllib.request.urlretrieve(_TINY_IN_URL, zip_path)
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


DATASET_REGISTRY = {
    "cifar10": {
        "cls": CIFAR10,
        "num_classes": 10,
        "image_size": 32,
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std":  (0.24703223, 0.24348513, 0.26158784),
    },
    "cifar100": {
        "cls": CIFAR100,
        "num_classes": 100,
        "image_size": 32,
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std":  (0.26733429, 0.25643846, 0.27615047),
    },
    "tiny_imagenet": {
        "cls": TinyImageNet,
        "num_classes": 200,
        "image_size": 64,
        "mean": (0.4802, 0.4481, 0.3975),
        "std":  (0.2302, 0.2265, 0.2262),
    },
}


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, return_numpy=False):
    collate_fn = numpy_collate_fn if return_numpy else None
    return {
        "train":      DataLoader(dataset["train"],      batch_size=batch_size, shuffle=shuffle,  drop_last=drop_last, collate_fn=collate_fn, num_workers=8, pin_memory=True),
        "validation": DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False,    drop_last=False,     collate_fn=collate_fn, num_workers=4, pin_memory=True),
        "test":       DataLoader(dataset["test"],       batch_size=512,        shuffle=False,    drop_last=False,     collate_fn=collate_fn, num_workers=4, pin_memory=True),
    }


def numpy_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0).numpy()
    labels = np.array([b[1] for b in batch], dtype=np.int32)
    return imgs, labels


def _build_transforms(meta, train: bool, augmentation: str = "simple"):
    size = meta["image_size"]
    if not train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ])
    if augmentation == "advanced":
        if size == 32:
            auto_aug = transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10)
        else:
            auto_aug = transforms.RandAugment(num_ops=2, magnitude=9)
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=size // 8),
            auto_aug,
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=meta["mean"]),
        ])
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(meta["mean"], meta["std"]),
    ])


def get_dataset(name, data_dir="data/", validation_size=5000, augmentation="simple"):
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    meta = DATASET_REGISTRY[name]
    cls = meta["cls"]

    train_transform = _build_transforms(meta, train=True, augmentation=augmentation)
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