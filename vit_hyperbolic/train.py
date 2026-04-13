import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict

from vit_hyperbolic.lorentz import Lorentz
from vit_hyperbolic.model import LorentzViT

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

    model_cfg = asdict(cfg)
    model_cfg["manifold"] = Lorentz(cfg.curvature)
    
    model = LorentzViT(model_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # warmup_epochs = 5
    # warmup = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    # )
    # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=cfg.epochs - warmup_epochs, eta_min=1e-6
    # )
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    # )

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model,
            trainloader,
            optimizer,
            loss_fn,
            device,
            detect_anomaly=cfg.detect_anomaly
        )
        # scheduler.step()
        train_acc = evaluate(model, trainloader, loss_fn, device)[0]
        acc, test_loss = evaluate(model, testloader, loss_fn, device)

        print(
            f"Epoch: {epoch + 1}, "
            f"Train loss: {train_loss:.4f}, "
            f"Train accuracy: {train_acc:.4f}, "
            f"Test loss: {test_loss:.4f}, "
            f"Test accuracy: {acc:.4f}"
        )

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()