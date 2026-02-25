import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hyperbolic.lorentz import Lorentz
from hyperbolic.model import LorentzViT


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

    # training (match train.py defaults/flags)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    # extra flags to mirror your non-wandb script
    parser.add_argument("--check_metrics", action="store_true", default=False)
    parser.add_argument("--check_caps", action="store_true", default=False)

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

    # wandb
    parser.add_argument("--wandb", type=int, default=1)  # 1 enable, 0 disable
    parser.add_argument("--wandb_project", type=str, default="hyperbolic-vit")
    parser.add_argument("--wandb_entity", type=str, default="hyperbolic-vit-team")
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.qkv_bias = bool(args.qkv_bias)
    args.wandb = bool(args.wandb)
    return args


def _accumulate_metrics(metric_sums, metric_counts, metrics: dict):
    # metrics: {stage: {k: float}}
    for stage, vals in metrics.items():
        for k, v in vals.items():
            key = f"{stage}.{k}"
            metric_sums[key] = metric_sums.get(key, 0.0) + float(v)
            metric_counts[key] = metric_counts.get(key, 0) + 1


def _mean_metrics(metric_sums, metric_counts):
    return {k: metric_sums[k] / metric_counts[k] for k in metric_sums}


def train_one_epoch(
    model,
    trainloader,
    optimizer,
    loss_fn,
    device,
    *,
    detect_anomaly=False,
    check_caps=False,
    check_metrics=False,
):
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
            _accumulate_metrics(metric_sums, metric_counts, metrics)

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits detected")

        loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(trainloader.dataset)

    epoch_metrics = _mean_metrics(metric_sums, metric_counts) if metric_sums else None

    cap_stats = None
    if check_caps:
        cap_stats = model.manifold.cap_stats()

    return epoch_loss, epoch_metrics, cap_stats


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

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )

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
        "final_op": args.final_op,
    }

    device = torch.device(args.device)
    model = LorentzViT(model_cfg).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, epoch_metrics, cap_stats = train_one_epoch(
            model,
            trainloader,
            optimizer,
            loss_fn,
            device,
            detect_anomaly=args.detect_anomaly,
            check_caps=args.check_caps,
            check_metrics=args.check_metrics,
        )
        acc, test_loss = evaluate(model, testloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1}, "
            f"Train loss: {train_loss:.4f}, "
            f"Test loss: {test_loss:.4f}, "
            f"Accuracy: {acc:.4f}"
        )

        if epoch_metrics is not None:
            print(f"epoch metrics: {epoch_metrics}")

        if cap_stats is not None:
            print(
                f"cap geo: {100*cap_stats['geo_frac']:.2f}% "
                f"({cap_stats['geo_hits']}/{cap_stats['geo_total']})"
            )

        if wandb_run is not None:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "test/loss": test_loss,
                "test/acc": acc,
            }

            # log the aggregated hyperbolic metrics
            if epoch_metrics is not None:
                for k, v in epoch_metrics.items():
                    log_dict[f"hyp/{k.replace('.', '/')}"] = v

                def _get(key):
                    return epoch_metrics.get(key, None)

                r_e = _get("after_expmap0.r_mean")
                r_enc = _get("after_encoder.r_mean")
                r_p = _get("after_pool.r_mean")
                if r_e is not None and r_enc is not None:
                    log_dict["hyp/dr_mean/encoder_minus_expmap0"] = r_enc - r_e
                if r_enc is not None and r_p is not None:
                    log_dict["hyp/dr_mean/pool_minus_encoder"] = r_p - r_enc

                rstd_e = _get("after_expmap0.r_std")
                rstd_enc = _get("after_encoder.r_std")
                if rstd_e is not None and rstd_enc is not None:
                    log_dict["hyp/dr_std/encoder_minus_expmap0"] = rstd_enc - rstd_e

            # log pre_scale if present on the model
            if hasattr(model, "pre_scale"):
                try:
                    log_dict["hyp/pre_scale"] = float(model.pre_scale.detach().cpu().item())
                except Exception:
                    pass

            # log cap stats
            if cap_stats is not None:
                log_dict["hyp/cap_geo_frac"] = float(cap_stats["geo_frac"])

            wandb_run.log(log_dict)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()