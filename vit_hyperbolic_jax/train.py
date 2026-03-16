"""
Training script for LorentzViT in JAX/Flax.

Usage:
    python -m vit_hyperbolic_jax.train [--epochs 30] [--lr 1e-3] ...

Requires:
    jax, flax, optax, cifar10_utils (same as original)
"""
from __future__ import annotations

import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import optax

from vit_hyperbolic_jax.model import LorentzViT
from vit_hyperbolic_jax.config import Config
from vit_hyperbolic_jax.utils import (
    build_parser,
    cfg_from_args,
    set_seed,
    create_train_state,
    train_one_epoch,
    evaluate,
)


def main():
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = cfg_from_args(args)
    print(cfg)

    key = set_seed(cfg.seed)

    run_start = time.perf_counter()

    from cifar10_utils import get_cifar10, get_dataloader
    dataset     = get_cifar10(data_dir="data/", validation_size=5000)
    dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)
    trainloader = dataloaders["train"]
    testloader  = dataloaders["test"]

    # Build model from config
    model_kwargs = asdict(cfg)
    # Remove training-only keys not accepted by LorentzViT
    for k in ("batch_size", "epochs", "lr", "weight_decay", "seed",
              "detect_anomaly", "check_metrics", "initializer_range", "qkv_bias"):
        model_kwargs.pop(k, None)

    model = LorentzViT(**model_kwargs)

    # Initialise parameters with a dummy batch
    key, init_key, dropout_key = jax.random.split(key, 3)
    dummy = jnp.zeros((1, cfg.num_channels, cfg.image_size, cfg.image_size))
    variables = model.init(
        {"params": init_key, "dropout": dropout_key},
        dummy,
        deterministic=True,
    )
    params = variables["params"]
    print(f"Initialized LorentzViT  |  params: {sum(p.size for p in jax.tree_util.tree_leaves(params)):,}")

    # Optimizer
    tx = optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)

    state = create_train_state(model, params, tx)

    for epoch in range(cfg.epochs):
        state, train_loss, key = train_one_epoch(state, trainloader, key)
        acc, test_loss         = evaluate(state, testloader)

        print(
            f"Epoch: {epoch + 1:3d}  "
            f"Train loss: {train_loss:.4f}  "
            f"Test loss: {test_loss:.4f}  "
            f"Accuracy: {acc:.4f}"
        )

    total_time = time.perf_counter() - run_start
    print(f"\nTotal training time: {total_time:.2f}s ({total_time / 60:.2f} min)")


if __name__ == "__main__":
    main()
