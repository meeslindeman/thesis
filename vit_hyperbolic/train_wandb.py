import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from dataclasses import asdict

from vit_hyperbolic.lorentz import Lorentz
from vit_hyperbolic.model import LorentzViT

from config import Config
from utils import build_parser, cfg_from_args, set_seed, evaluate


def train_one_epoch(model, trainloader, optimizer, loss_fn, device, *, detect_anomaly: bool = False, check_metrics: bool = False):
    """Like utils.train_one_epoch, but returns (avg_loss, metric_means) so they can be logged."""
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

        if metrics is not None:
            for stage, vals in metrics.items():
                for k, v in vals.items():
                    key = f"{stage}.{k}"
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(v)
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits detected")

        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    metric_means = {k: metric_sums[k] / metric_counts[k] for k in metric_sums}
    return total_loss / len(trainloader.dataset), metric_means


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    print(cfg)

    set_seed(cfg.seed)

    wandb.init(
        entity="hyperbolic-vit-team",
        project="hyperbolic-vit",
        config=asdict(cfg),
    )

    run_start = time.perf_counter()

    from cifar10_utils import get_cifar10, get_dataloader
    dataset = get_cifar10(data_dir="data/", validation_size=5000)
    dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)
    trainloader = dataloaders["train"]
    testloader = dataloaders["test"]

    device = torch.device(cfg.device)

    model_cfg = asdict(cfg)
    model_cfg["manifold"] = Lorentz(cfg.curvature)

    model = LorentzViT(model_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        train_loss, epoch_metrics = train_one_epoch(
            model,
            trainloader,
            optimizer,
            loss_fn,
            device,
            detect_anomaly=cfg.detect_anomaly,
            check_metrics=cfg.check_metrics,
        )
        acc, test_loss = evaluate(model, testloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1}, "
            f"Train loss: {train_loss:.4f}, "
            f"Test loss: {test_loss:.4f}, "
            f"Accuracy: {acc:.4f}"
        )

        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "test/loss": test_loss,
            "test/accuracy": acc,
        }
        if epoch_metrics:
            for k, v in epoch_metrics.items():
                log_dict[f"metrics/{k}"] = v

        wandb.log(log_dict, step=epoch + 1)

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    wandb.log({"train/total_time_seconds": total_time})

    wandb.finish()


if __name__ == "__main__":
    main()
