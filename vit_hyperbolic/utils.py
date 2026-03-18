import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from dataclasses import fields
from config import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser(defaults: Config | None = None) -> argparse.ArgumentParser:
    """
    Builds argparse parser from Config fields.
    If defaults is provided, those are used as argparse defaults.
    """
    defaults = defaults or Config()
    p = argparse.ArgumentParser()

    for f in fields(Config):
        name = f.name
        t = f.type
        default_val = getattr(defaults, name)

        if t is bool:
            # --flag / --no-flag
            group = p.add_mutually_exclusive_group(required=False)
            group.add_argument(f"--{name}", dest=name, action="store_true")
            group.add_argument(f"--no-{name}", dest=name, action="store_false")
            p.set_defaults(**{name: default_val})
        else:
            # allow passing None for device by omitting
            p.add_argument(f"--{name}", type=t if t is not None else str, default=default_val)

    return p


def cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    for f in fields(Config):
        setattr(cfg, f.name, getattr(args, f.name))
    # normalize device
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def train_one_epoch(model, trainloader, optimizer, loss_fn, device, *, detect_anomaly: bool = False, check_metrics: bool = False) -> float:

    model.train()
    total_loss = 0.0

    metric_sums = {}
    metric_counts = {}

    torch.autograd.set_detect_anomaly(detect_anomaly)

    for images, labels in trainloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        logits, metrics = model(images, return_metrics=check_metrics)

        if logits.isnan().any() or logits.isinf().any():
            raise ValueError("Logits contain NaN or Inf values.")

        if metrics is not None:
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                metric_counts[k] = metric_counts.get(k, 0) + 1

        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    if metric_sums:
        metric_means = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        print(f"epoch metrics: {metric_means}")

    return total_loss / len(trainloader.dataset)


@torch.no_grad()
def evaluate(model, testloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    for images, labels in testloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        logits, _ = model(images, return_metrics=False)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    acc = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader.dataset)
    return acc, avg_loss