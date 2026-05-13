import argparse
import random
from pathlib import Path

import numpy as np
import torch
from dataclasses import fields


def _make_scheduler(cfg, opt):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    elif cfg.scheduler == "warmup_cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=cfg.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
        )
    return None


def build_scheduler(cfg, optimizer):
    if hasattr(optimizer, "optimizers"):
        schedulers = [s for o in optimizer.optimizers if (s := _make_scheduler(cfg, o)) is not None]
        if not schedulers:
            return None

        class DualScheduler:
            def step(self):
                for s in schedulers:
                    s.step()

        return DualScheduler()

    return _make_scheduler(cfg, optimizer)


# Seed setting and device management
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    return batch


# Checkpointing
def save_checkpoint(path, model, optimizer=None, scheduler=None, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), **extra}
    if optimizer is not None:
        if hasattr(optimizer, "optimizers"):
            state["optimizer"] = [o.state_dict() for o in optimizer.optimizers]
        else:
            state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        if hasattr(scheduler, "schedulers"):
            state["scheduler"] = [s.state_dict() for s in scheduler.schedulers]
        else:
            state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        opt_state = state["optimizer"]
        if hasattr(optimizer, "optimizers") and isinstance(opt_state, list):
            for o, s in zip(optimizer.optimizers, opt_state):
                o.load_state_dict(s)
        else:
            optimizer.load_state_dict(opt_state)
    if scheduler is not None and "scheduler" in state:
        sched_state = state["scheduler"]
        if hasattr(scheduler, "schedulers") and isinstance(sched_state, list):
            for s, sd in zip(scheduler.schedulers, sched_state):
                s.load_state_dict(sd)
        else:
            scheduler.load_state_dict(sched_state)
    return state


# Configuration and argument parsing
def build_parser(Config) -> argparse.ArgumentParser:
    defaults = Config()
    p = argparse.ArgumentParser()
    for f in fields(Config):
        if not f.init:
            continue
        name, t, default_val = f.name, f.type, getattr(defaults, f.name)
        if t is bool:
            group = p.add_mutually_exclusive_group(required=False)
            group.add_argument(f"--{name}", dest=name, action="store_true")
            group.add_argument(f"--no-{name}", dest=name, action="store_false")
            p.set_defaults(**{name: default_val})
        else:
            p.add_argument(f"--{name}", type=t if t is not None else str, default=default_val)
    return p


def cfg_from_args(args: argparse.Namespace, Config):
    init_kwargs = {f.name: getattr(args, f.name) for f in fields(Config) if f.init}
    cfg = Config(**init_kwargs)
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


# Training / evaluation loops
def train_one_epoch(model, trainloader, optimizer, loss_fn, device, grad_clip: float = 0.0) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    for images, labels in trainloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
    n = len(trainloader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return correct / n, total_loss / n
