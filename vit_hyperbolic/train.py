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

    set_seed(cfg.seed)

    run_start = time.perf_counter()

    from cifar10_utils import get_cifar10, get_dataloader
    dataset = get_cifar10(data_dir="data/", validation_size=5000)
    dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)
    trainloader = dataloaders["train"]
    testloader = dataloaders["test"]

    device = torch.device(cfg.device)

    # model_cfg = {
    #     "patch_size": cfg.patch_size,
    #     "hidden_size": cfg.hidden_size,
    #     "num_hidden_layers": cfg.num_hidden_layers,
    #     "num_attention_heads": cfg.num_attention_heads,
    #     "mlp_dim": cfg.mlp_dim,
    #     "dropout_rate": cfg.dropout_rate,
    #     "initializer_range": cfg.initializer_range,
    #     "image_size": cfg.image_size,
    #     "num_classes": cfg.num_classes,
    #     "num_channels": cfg.num_channels,
    #     "qkv_bias": cfg.qkv_bias,
    #     "curvature": cfg.curvature,
    #     "manifold": Lorentz(cfg.curvature),
    #     "head_mode": cfg.head_mode,
    #     "concat_mode": cfg.concat_mode,
    #     "compute_scores": cfg.compute_scores,
    #     "reset_params": cfg.reset_params,
    #     "a_default": cfg.a_default,
    #     "skip_connection": cfg.skip_connection,
    #     "final_op": cfg.final_op,
    #     "embeddings": cfg.embeddings,
    #     "patch_embed_type": cfg.patch_embed_type,
    # }

    model_cfg = asdict(cfg)
    model_cfg["manifold"] = Lorentz(cfg.curvature)
    
    model = LorentzViT(model_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
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

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()