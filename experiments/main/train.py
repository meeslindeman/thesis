import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict

from pathlib import Path

from data_utils import get_dataset, get_dataloader
from utils import build_parser, cfg_from_args, load_yaml, set_seed, train_one_epoch, evaluate, save_checkpoint, load_checkpoint, build_scheduler

MODELS = ["euclidean", "hyperbolic", "hypercore"]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIRS = {
    "euclidean": REPO_ROOT / "vit",
    "hyperbolic": REPO_ROOT / "vit_hyperbolic",
    "hypercore":  REPO_ROOT / "vit_hypercore",
}


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
        from vit_hypercore.model import LViT
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
            hidden_channel=cfg.hidden_size // cfg.num_attention_heads,
            out_channel=cfg.num_classes,
            mlp_hidden_size=cfg.mlp_dim // cfg.num_attention_heads,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout_rate,
        )


def _hyperbolic_param_groups(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(('.U', '.v')):
            decay.append(param)
        else:
            no_decay.append(param)
    return optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr,
    )


def build_optimizer(model_name: str, model, cfg):
    if model_name == "euclidean":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if model_name == "hyperbolic":
        if cfg.optimizer == "riemannian":
            from geoopt.optim import RiemannianAdam
            return RiemannianAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        return _hyperbolic_param_groups(model, cfg.lr, cfg.weight_decay)

    if model_name == "hypercore":
        from vit_hypercore.optimizer import build_optimizer as _build
        return _build(model, cfg)


def main():
    # Phase 1: extract --model before building the full parser, so each model's
    # config fields are correctly registered (and --help shows the right ones).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="hyperbolic", choices=MODELS)
    pre_args, _ = pre.parse_known_args()

    # Phase 2: load model YAML, then build parser with those as defaults.
    from config_base import Config
    yaml_overrides = load_yaml(MODEL_DIRS[pre_args.model] / "config.yml")
    parser = build_parser(Config, yaml_overrides)
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

    dataset, _ = get_dataset(cfg.dataset, data_dir="data/", validation_size=5000, augmentation=cfg.augmentation)
    dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)
    trainloader = dataloaders["train"]
    valloader = dataloaders["validation"]
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
    best_val_acc = 0.0
    best_test_acc = 0.0
    epochs_no_improve = 0

    ckpt_path = None
    if args.checkpoint_dir is not None:
        ckpt_path = Path(args.checkpoint_dir) / f"{args.model}_{cfg.dataset}.pt"

    if args.resume is not None:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = state.get("epoch", 0)
        best_val_acc = state.get("val_acc", 0.0)
        best_test_acc = state.get("test_acc", 0.0)
        print(f"Resumed from epoch {start_epoch} (val_acc={best_val_acc:.4f}, test_acc={best_test_acc:.4f})")

    for epoch in range(start_epoch, cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, loss_fn, device, grad_clip=cfg.grad_clip, mixing_aug=(cfg.augmentation == "advanced"))
        if scheduler is not None:
            scheduler.step()
        val_acc, val_loss = evaluate(model, valloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1}, "
            f"Train loss: {train_loss:.4f}, "
            f"Train accuracy: {train_acc:.4f}, "
            f"Val loss: {val_loss:.4f}, "
            f"Val accuracy: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc, _ = evaluate(model, testloader, loss_fn, device)
            epochs_no_improve = 0
            if ckpt_path is not None:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=epoch + 1, val_acc=best_val_acc, test_acc=best_test_acc, model_name=args.model)
                print(f"Saved checkpoint: {ckpt_path} (val={best_val_acc:.4f}, test={best_test_acc:.4f})")
        else:
            epochs_no_improve += 1

        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {cfg.patience} epochs)")
            break

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.4f} — Test accuracy at best val: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
