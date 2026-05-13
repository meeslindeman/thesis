import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict

from pathlib import Path

from data_utils import get_dataset, get_dataloader
from utils import build_parser, cfg_from_args, set_seed, train_one_epoch, evaluate, save_checkpoint, load_checkpoint, build_scheduler

MODELS = ["euclidean", "hyperbolic", "hypercore"]


def get_config_class(model: str):
    if model == "euclidean":
        from vit.config import Config
    elif model == "hyperbolic":
        from vit_hyperbolic.config import Config
    elif model == "hypercore":
        from vit_hypercore.config import Config
    else:
        raise ValueError(f"Unknown model '{model}'. Choose from: {MODELS}")
    return Config


def build_model(model: str, cfg):
    if model == "euclidean":
        from vit.model import ViT
        return ViT(asdict(cfg))

    elif model == "hyperbolic":
        from vit_hyperbolic.lorentz import Lorentz
        from vit_hyperbolic.model import LorentzViT
        model_cfg = asdict(cfg)
        model_cfg["manifold"] = Lorentz(cfg.curvature)
        return LorentzViT(model_cfg)

    elif model == "hypercore":
        from vit_hypercore.LViT import LViT
        from vit_hypercore.manifold.lorentz import Lorentz
        manifold = Lorentz(cfg.curvature)
        return LViT(
            manifold_in=manifold,
            manifold_hidden=manifold,
            manifold_out=manifold,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_hidden_layers,
            in_channel=cfg.num_channels,
            hidden_channel=cfg.hidden_channel,
            out_channel=cfg.num_classes,
            mlp_hidden_size=cfg.mlp_hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout_rate,
        )


def build_optimizer(model_name: str, model, cfg):
    if model_name == "euclidean":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if model_name == "hyperbolic":
        if cfg.optimizer == "riemannian":
            from geoopt.optim import RiemannianAdam
            return RiemannianAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if model_name == "hypercore":
        from vit_hypercore.optimizer import build_optimizer as _build
        return _build(model, cfg)


def main():
    # Phase 1: extract --model before building the full parser, so each model's
    # config fields are correctly registered (and --help shows the right ones).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="hyperbolic", choices=MODELS)
    pre_args, _ = pre.parse_known_args()

    # Phase 2: build the full parser from the selected model's Config class.
    Config = get_config_class(pre_args.model)
    parser = build_parser(Config)
    parser.add_argument("--model", default=pre_args.model, choices=MODELS,
                        help="Which model to train (euclidean | hyperbolic | hypercore)")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory to save checkpoints (omit to skip saving)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    cfg = cfg_from_args(args, Config)

    print(f"Model: {args.model}")
    print(cfg)

    set_seed(cfg.seed)
    run_start = time.perf_counter()

    dataset, _ = get_dataset(cfg.dataset, data_dir="data/", validation_size=5000)
    dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)
    trainloader = dataloaders["train"]
    testloader = dataloaders["test"]

    device = torch.device(cfg.device)
    model = build_model(args.model, cfg).to(device)

    # count actually-used parameters via a dummy forward/backward pass
    _dummy = torch.zeros(1, cfg.num_channels, cfg.image_size, cfg.image_size, device=device)
    model(_dummy).sum().backward()
    total_params = sum(p.numel() for p in model.parameters())
    used_params = sum(p.numel() for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    print(f"Parameters: {total_params:,} total, {used_params:,} used")

    optimizer = build_optimizer(args.model, model, cfg)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 0
    best_acc = 0.0
    epochs_no_improve = 0

    if args.resume is not None:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = state.get("epoch", 0)
        best_acc = state.get("acc", 0.0)
        print(f"Resumed from epoch {start_epoch} (acc={best_acc:.4f})")

    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, loss_fn, device, grad_clip=cfg.grad_clip)
        if scheduler is not None:
            scheduler.step()
        acc, test_loss = evaluate(model, testloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1}, "
            f"Train loss: {train_loss:.4f}, "
            f"Train accuracy: {train_acc:.4f}, "
            f"Test loss: {test_loss:.4f}, "
            f"Test accuracy: {acc:.4f}"
        )

        if cfg.patience > 0:
            if acc > best_acc:
                best_acc = acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {cfg.patience} epochs)")
                break

    if args.checkpoint_dir is not None:
        ckpt_path = Path(args.checkpoint_dir) / f"{args.model}_{cfg.dataset}.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=epoch + 1, acc=acc, model_name=args.model)
        print(f"Saved checkpoint: {ckpt_path}")

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
