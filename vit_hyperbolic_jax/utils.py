from __future__ import annotations

import argparse
import random
import numpy as np
from dataclasses import fields
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from vit_hyperbolic_jax.config import Config


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> jax.Array:
    """Set Python/NumPy seeds and return a JAX PRNG key."""
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser(defaults: Config | None = None) -> argparse.ArgumentParser:
    """Build argparse parser from Config fields."""
    defaults = defaults or Config()
    p = argparse.ArgumentParser()
    for f in fields(Config):
        name = f.name
        t = f.type
        default_val = getattr(defaults, name)
        if t is bool:
            group = p.add_mutually_exclusive_group(required=False)
            group.add_argument(f"--{name}", dest=name, action="store_true")
            group.add_argument(f"--no-{name}", dest=name, action="store_false")
            p.set_defaults(**{name: default_val})
        else:
            p.add_argument(f"--{name}", type=t if t is not None else str, default=default_val)
    return p


def cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    for f in fields(Config):
        setattr(cfg, f.name, getattr(args, f.name))
    return cfg


# ---------------------------------------------------------------------------
# Training state helpers (Flax / Optax)
# ---------------------------------------------------------------------------

def create_train_state(model, params, tx):
    """Create a simple training state container."""
    from flax.training import train_state
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# ---------------------------------------------------------------------------
# JIT-compiled train and eval steps
# ---------------------------------------------------------------------------

@jax.jit
def train_step(state, images: jnp.ndarray, labels: jnp.ndarray, dropout_key: jax.Array) -> Tuple[Any, float]:
    """Single gradient update step."""

    def loss_fn(params):
        logits, _ = state.apply_fn(
            {"params": params},
            images,
            deterministic=False,
            rngs={"dropout": dropout_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    has_nan = jnp.any(jnp.isnan(logits))
    return state, loss, has_nan


@jax.jit
def eval_step(state, images: jnp.ndarray, labels: jnp.ndarray) -> Tuple[float, float]:
    """Single evaluation step; returns (loss, num_correct)."""
    logits, _ = state.apply_fn(
        {"params": state.params},
        images,
        deterministic=True,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    num_correct = (jnp.argmax(logits, axis=-1) == labels).sum()
    return loss, num_correct


# ---------------------------------------------------------------------------
# Epoch-level loops
# ---------------------------------------------------------------------------

def train_one_epoch(state, trainloader, key: jax.Array) -> Tuple[Any, float, jax.Array]:
    """Run one training epoch. Returns (new_state, avg_loss, new_key)."""
    total_loss  = 0.0
    total_count = 0

    for images, labels in trainloader:
        images = jnp.array(images)
        labels = jnp.array(labels)

        key, dropout_key = jax.random.split(key)
        state, loss, has_nan = train_step(state, images, labels, dropout_key)

        if has_nan:
            raise ValueError(f"NaN in logits detected after {total_count} samples — aborting training.")
        
        if jnp.isnan(loss):
            raise ValueError(f"NaN loss detected after {total_count} samples — aborting training.")

        total_loss += float(loss) * images.shape[0]
        total_count += images.shape[0]

    return state, total_loss / total_count, key


def evaluate(state, testloader) -> Tuple[float, float]:
    """Evaluate over the full test set. Returns (accuracy, avg_loss)."""
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in testloader:
        images = jnp.array(images)
        labels = jnp.array(labels)

        loss, num_correct = eval_step(state, images, labels)

        total_loss += float(loss) * images.shape[0]
        total_correct += int(num_correct)
        total_count += images.shape[0]

    acc = total_correct / total_count
    avg_loss = total_loss / total_count
    return acc, avg_loss
