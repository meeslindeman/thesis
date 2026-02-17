import torch
import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, return_numpy=False):
    collate_fn = numpy_collate_fn if return_numpy else None
    train_dataloader = DataLoader(dataset=dataset["train"], batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)
    validation_dataloader = DataLoader(dataset=dataset["validation"], batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=dataset["test"], batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    return {"train": train_dataloader, "validation": validation_dataloader, "test": test_dataloader}


def numpy_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0).numpy()
    labels = np.array([b[1] for b in batch], dtype=np.int32)
    return imgs, labels


def read_data_sets(data_dir, validation_size=5000):
    mean = (0.49139968, 0.48215841, 0.44653091)
    std  = (0.24703223, 0.24348513, 0.26158784)

    data_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])

    train_dataset = CIFAR10(root=data_dir, train=True, download=False, transform=data_transforms)
    test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=data_transforms)

    # Subsample the validation set from the train set
    if not 0 <= validation_size <= len(train_dataset):
        raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
            len(train_dataset), validation_size))

    train_dataset, validation_dataset = random_split(train_dataset,
                                                     lengths=[len(train_dataset) - validation_size, validation_size],
                                                     generator=torch.Generator().manual_seed(42))

    return {'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset}


def get_cifar10(data_dir='data/', validation_size=5000):
    return read_data_sets(data_dir, validation_size)
