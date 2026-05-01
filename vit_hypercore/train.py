import time
import torch
import torch.nn as nn
import torch.optim as optim

from vit_hypercore.LViT import LViT
from vit_hypercore.manifold.lorentz import Lorentz

from config import Config
from utils import build_parser, cfg_from_args, set_seed, train_one_epoch, evaluate


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    print(cfg)

    set_seed(cfg.seed)

    run_start = time.perf_counter()

    from data_utils import get_dataset, get_dataloader

    dataset, _ = get_dataset(cfg.dataset, data_dir="data/", validation_size=5000)
    dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)

    trainloader = dataloaders["train"]
    testloader = dataloaders["test"]

    device = torch.device(cfg.device)

    manifold = Lorentz(cfg.curvature)
    model = LViT(
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
    ).to(device)

    _dummy = torch.zeros(1, cfg.num_channels, cfg.image_size, cfg.image_size, device=device)
    model(_dummy).sum().backward()
    total_params = sum(p.numel() for p in model.parameters())
    used_params  = sum(p.numel() for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    print(f"Parameters: {total_params:,} total, {used_params:,} used")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )
    elif cfg.scheduler == "warmup_cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=cfg.warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
        )
    else:
        scheduler = None

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, loss_fn, device)
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

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
