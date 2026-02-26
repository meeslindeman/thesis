import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hyperbolic.lorentz import Lorentz
from hyperbolic.model import LorentzViT

import traceback
from contextlib import contextmanager

def _iter_tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _iter_tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _iter_tensors(y)

class FirstNonFiniteError(RuntimeError):
    pass

def register_first_nonfinite_hooks(
    model: nn.Module,
    *,
    include_backward: bool = False,
    only_modules_with_params: bool = False,
    ignore_modules: tuple[type, ...] = (),
):
    """
    Registers hooks on modules to catch the first NaN/Inf in forward (and optionally backward).
    Raises immediately with module name/type and tensor stats.
    """
    handles = []
    tripped = {"flag": False}

    def should_hook(m: nn.Module) -> bool:
        if ignore_modules and isinstance(m, ignore_modules):
            return False
        if only_modules_with_params:
            return any(p.requires_grad for p in m.parameters(recurse=False))
        return True

    def forward_hook(name):
        def hook(mod, inp, out):
            if tripped["flag"]:
                return
            for t in _iter_tensors(out):
                if t.numel() == 0:
                    continue
                if not torch.isfinite(t).all():
                    tripped["flag"] = True
                    # summarize without crashing on NaNs
                    finite = t[torch.isfinite(t)]
                    fin_min = float(finite.min()) if finite.numel() else float("nan")
                    fin_max = float(finite.max()) if finite.numel() else float("nan")
                    msg = (
                        f"\nFIRST NON-FINITE in forward\n"
                        f"module: {name} ({mod.__class__.__name__})\n"
                        f"output: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}\n"
                        f"finite_min={fin_min} finite_max={fin_max}\n"
                        f"nan_count={(~torch.isfinite(t)).sum().item()}\n"
                    )
                    print(msg)
                    print("Python stack (most recent call last):")
                    traceback.print_stack(limit=30)
                    raise FirstNonFiniteError(msg)
        return hook

    def backward_hook(name):
        def hook(mod, grad_in, grad_out):
            if tripped["flag"]:
                return
            for t in _iter_tensors(grad_out):
                if t is None or (torch.is_tensor(t) and t.numel() == 0):
                    continue
                if torch.is_tensor(t) and (not torch.isfinite(t).all()):
                    tripped["flag"] = True
                    finite = t[torch.isfinite(t)]
                    fin_min = float(finite.min()) if finite.numel() else float("nan")
                    fin_max = float(finite.max()) if finite.numel() else float("nan")
                    msg = (
                        f"\nFIRST NON-FINITE in backward\n"
                        f"module: {name} ({mod.__class__.__name__})\n"
                        f"grad_out: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}\n"
                        f"finite_min={fin_min} finite_max={fin_max}\n"
                        f"nan_count={(~torch.isfinite(t)).sum().item()}\n"
                    )
                    print(msg)
                    print("Python stack (most recent call last):")
                    traceback.print_stack(limit=30)
                    raise FirstNonFiniteError(msg)
        return hook

    for name, m in model.named_modules():
        if not should_hook(m):
            continue
        handles.append(m.register_forward_hook(forward_hook(name)))
        if include_backward:
            # full backward hook is safer than legacy backward_hook
            handles.append(m.register_full_backward_hook(backward_hook(name)))

    return handles

@contextmanager
def hook_context(handles):
    try:
        yield
    finally:
        for h in handles:
            h.remove()

# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--check_caps", action="store_true", default=False)
    parser.add_argument("--check_metrics", action="store_true", default=False)

    # model (LorentzViT)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=48)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=192)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--qkv_bias", type=int, default=1)  # 0/1 -> bool
    parser.add_argument("--final_op", type=str, default="cls")  # "cls" or "midpoint"    

    # hyperbolic specifics
    parser.add_argument("--curvature", type=float, default=0.1)
    parser.add_argument("--head_mode", type=str, default="split_concat")
    parser.add_argument("--concat_mode", type=str, default="direct")
    parser.add_argument("--compute_scores", type=str, default="lorentz_inner")
    parser.add_argument("--reset_params", type=str, default="lorentz_kaiming")
    parser.add_argument("--skip_connection", action="store_true", default=False)
    parser.add_argument("--a_default", type=float, default=0.0)

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.qkv_bias = bool(args.qkv_bias)
    return args


def train_one_epoch(model, trainloader, optimizer, loss_fn, device, *, detect_anomaly=False, check_caps=False, check_metrics=False):
    if check_caps:
        model.manifold.reset_cap_stats()

    model.train()
    total_loss = 0.0

    metric_sums = {}
    metric_counts = {}

    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    for images, labels in trainloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        logits, metrics = model(images, return_metrics=check_metrics)

        if metrics is not None:
            for stage, vals in metrics.items():
                for k, v in vals.items():
                    key = f"{stage}.{k}"
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(v)
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        # Safety check
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits detected")

        loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    if metric_sums:
        metric_means = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
        print(f"epoch metrics: {metric_means}")
    
    if check_caps:
        stats = model.manifold.cap_stats()
        print(
            f"cap geo: {100*stats['geo_frac']:.2f}% "
            f"({stats['geo_hits']}/{stats['geo_total']})"
        )

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


def main():
    args = parse_args()
    print(args)

    set_seed(args.seed)

    from cifar10_utils import get_cifar10, get_dataloader

    dataset = get_cifar10(data_dir="data/", validation_size=5000)
    dataloaders = get_dataloader(dataset, args.batch_size, shuffle=True, drop_last=True)
    trainloader = dataloaders["train"]
    testloader = dataloaders["test"]

    model_cfg = {
        "patch_size": args.patch_size,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "mlp_dim": args.mlp_dim,
        "dropout_rate": args.dropout_rate,
        "initializer_range": args.initializer_range,
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "num_channels": args.num_channels,
        "qkv_bias": args.qkv_bias,
        "curvature": args.curvature,
        "manifold": Lorentz(args.curvature),
        "head_mode": args.head_mode,
        "concat_mode": args.concat_mode,
        "compute_scores": args.compute_scores,
        "reset_params": args.reset_params,
        "a_default": args.a_default,
        "skip_connection": args.skip_connection,
        "final_op": args.final_op
    }

    device = torch.device(args.device)
    model = LorentzViT(model_cfg).to(device)


    handles = register_first_nonfinite_hooks(
        model,
        include_backward=False,          # set True if you also suspect grads
        only_modules_with_params=False,  # set True to reduce noise
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            trainloader,
            optimizer,
            loss_fn,
            device,
            detect_anomaly=args.detect_anomaly,
            check_caps=args.check_caps,
            check_metrics=args.check_metrics
        )
        acc, test_loss = evaluate(model, testloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1}, "
            f"Train loss: {train_loss:.4f}, "
            f"Test loss: {test_loss:.4f}, "
            f"Accuracy: {acc:.4f}"
        )


if __name__ == "__main__":
    main()
