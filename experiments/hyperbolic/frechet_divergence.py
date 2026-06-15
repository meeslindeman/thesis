"""
Empirical comparison: Lorentzian centroid vs. true Fréchet mean under softmax attention.

For each attention layer, hooks into the lorentz_centroid call to capture value vectors
and attention weights, then computes the true Fréchet mean via Riemannian gradient descent
and measures the geodesic distance to the centroid.

The Fréchet mean is computed in a fully vectorised manner over all (batch, head) pairs
and sampled query tokens simultaneously, making GPU execution practical.

Usage:
    python frechet_divergence.py [--checkpoint PATH] [--n_batches 5]
                                  [--batch_size 32] [--sample_tokens 8]
                                  [--n_iter 100] [--lr 0.3]
                                  [--device cuda] [--out results_frechet.json]
"""

import sys
import json
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "main"))


# ── Fréchet mean via Riemannian GD (fully batched) ──────────────────────────

def frechet_means_batched(
    manifold,
    points_bhnd: torch.Tensor,   # [BH, N, D]  — values per (b, h)
    weights_bhqn: torch.Tensor,  # [BH, Q, N]  — softmax weights per (b, h, query)
    n_iter: int = 100,
    lr: float = 0.3,
) -> torch.Tensor:               # [BH, Q, D]
    """
    Fully vectorised Riemannian GD for the Fréchet mean.

    Minimises F(μ) = Σ_j w_j d(μ, v_j)²  for each (b,h,query) simultaneously.
    Gradient: ∇F(μ) = -2 Σ_j w_j Log_μ(v_j)
    """
    k = manifold.k().to(points_bhnd)
    eps = manifold.eps
    BH, N, D = points_bhnd.shape
    Q = weights_bhqn.shape[1]

    # Initialise with the Lorentzian centroid
    # lorentz_centroid einsum "...qk,...kd->...qd":
    #   xs=[BH, N, D] (...=BH, k=N, d=D), w=[BH, Q, N] (...=BH, q=Q, k=N) → [BH, Q, D]
    with torch.no_grad():
        mu = manifold.lorentz_centroid(points_bhnd, weights_bhqn).clone()  # [BH, Q, D]

        sqrt_k = torch.sqrt(k)

        for _ in range(n_iter):
            # Log_{mu[bh,q]}(v[bh,j]) for all (bh, q, j) → [BH, Q, N, D]
            mu_e = mu.unsqueeze(2)           # [BH, Q, 1, D]
            pts_e = points_bhnd.unsqueeze(1) # [BH, 1, N, D]

            # Lorentz inner product <mu_e, pts_e>_L → [BH, Q, N, 1]
            xy = (-mu_e[..., :1] * pts_e[..., :1]
                  + (mu_e[..., 1:] * pts_e[..., 1:]).sum(-1, keepdim=True))

            alpha = (-k * xy).clamp_min(1.0 + eps)
            ac = torch.acosh(alpha)
            denom = torch.sqrt((alpha * alpha - 1).clamp_min(0.0))
            factor = torch.where(denom < eps, torch.ones_like(denom), ac / denom)

            u = pts_e + (k * xy) * mu_e     # [BH, Q, N, D]
            logs = factor * u               # Log_mu(v): [BH, Q, N, D]

            # Riemannian gradient: -2 * Σ_j w_j Log_mu(v_j)
            # weights_bhqn: [BH, Q, N] → unsqueeze → [BH, Q, N, 1]
            grad = -2.0 * (weights_bhqn.unsqueeze(-1) * logs).sum(2)  # [BH, Q, D]

            # Geodesic step: mu ← Exp_mu(-lr * grad)
            step = -lr * grad
            # Minkowski norm² of step (must be ≥ 0 for a spacelike tangent)
            vv = (-step[..., :1] * step[..., :1]
                  + (step[..., 1:] * step[..., 1:]).sum(-1, keepdim=True)).clamp_min(1e-8)
            norm_v = torch.sqrt(vv)
            arg = (sqrt_k * norm_v).clamp(max=85.0)
            left = torch.cosh(arg)
            right = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)
            mu_new = left * mu + right * step

            # Project back to hyperboloid (adjust time coordinate)
            space = mu_new[..., 1:]
            time = torch.sqrt((1.0 / k + (space * space).sum(-1, keepdim=True)).clamp_min(eps))
            mu = torch.cat([time, space], dim=-1)

    return mu  # [BH, Q, D]


# ── Experiment ───────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model      : {ck['model_name']}  val_acc={ck['val_acc']:.4f}  dataset={cfg['dataset']}")

    from vit_hyperbolic.lorentz import Lorentz
    from vit_hyperbolic.model import LorentzViT

    manifold = Lorentz(cfg["curvature"])
    cfg["manifold"] = manifold
    model = LorentzViT(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    n_layers = len(model.encoder.blocks)
    print(f"Layers     : {n_layers}  heads={cfg['num_attention_heads']}")

    from data_utils import get_dataset, get_dataloader
    datasets, _ = get_dataset(cfg["dataset"], augmentation="basic")
    loaders = get_dataloader(datasets, batch_size=args.batch_size)
    val_loader = loaders["validation"]

    # Monkey-patch lorentz_centroid on each attention layer's own manifold to capture (v, attn).
    # The hook signature matches the actual call: lorentz_centroid(v, attn)
    #   v   : [B, H, N, D_head]
    #   attn: [B, H, N, N]      — attn[b, h, q, :] = softmax weights for query q
    layer_captures = {i: [] for i in range(n_layers)}
    original_centroids = {}

    for layer_idx, block in enumerate(model.encoder.blocks):
        attn_manifold = block.attention.manifold
        orig = attn_manifold.lorentz_centroid
        original_centroids[layer_idx] = orig

        def _make_wrapper(idx, orig_fn):
            def wrapper(xs, w):
                result = orig_fn(xs, w)
                layer_captures[idx].append((xs.detach().cpu(), w.detach().cpu()))
                return result
            return wrapper

        attn_manifold.lorentz_centroid = _make_wrapper(layer_idx, orig)

    print(f"\nRunning {args.n_batches} validation batch(es) …")
    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(val_loader):
            if batch_idx >= args.n_batches:
                break
            model(imgs.to(device))
            print(f"  batch {batch_idx + 1}/{args.n_batches} done")

    for layer_idx, block in enumerate(model.encoder.blocks):
        block.attention.manifold.lorentz_centroid = original_centroids[layer_idx]

    eval_manifold = Lorentz(cfg["curvature"])
    k = eval_manifold.k()
    eps = eval_manifold.eps
    rng = torch.Generator()
    rng.manual_seed(42)

    print(f"\nComputing Fréchet means  n_iter={args.n_iter}  lr={args.lr}  "
          f"sample_tokens={args.sample_tokens} …")

    aggregate_dists = []
    aggregate_ratios = []
    aggregate_entropies = []
    results = {}

    for layer_idx in range(n_layers):
        records = layer_captures[layer_idx]
        if not records:
            continue

        layer_dists = []
        layer_ratios = []
        layer_entropies = []

        for batch_v, batch_attn in records:
            # batch_v   : [B, H, N, D_head]
            # batch_attn: [B, H, N, N]
            B, H, N, D = batch_v.shape

            # Flatten (B, H) into a single BH dimension
            v_bh = batch_v.reshape(B * H, N, D).to(device)     # [BH, N, D]
            a_bh = batch_attn.reshape(B * H, N, N).to(device)  # [BH, N, N]

            # Sample Q query token positions (same indices for all (b,h) in this batch)
            Q = min(args.sample_tokens, N) if args.sample_tokens else N
            q_idx = torch.randperm(N, generator=rng)[:Q]

            # weights_bhqn: [BH, Q, N]
            w_bhqn = a_bh[:, q_idx, :]  # [BH, Q, N]

            # Shannon entropy of each query's attention distribution: H = -Σ α log α
            entropy = -(w_bhqn * w_bhqn.clamp_min(1e-9).log()).sum(-1)  # [BH, Q]

            # Centroid for each (bh, q): lorentz_centroid(xs=[BH,N,D], w=[BH,Q,N]) → [BH,Q,D]
            centroid = eval_manifold.lorentz_centroid(v_bh, w_bhqn)  # [BH, Q, D]

            # True Fréchet mean
            frechet = frechet_means_batched(
                eval_manifold, v_bh, w_bhqn, n_iter=args.n_iter, lr=args.lr
            )  # [BH, Q, D]

            # Geodesic distance between centroid and Fréchet mean: [BH, Q]
            ip = eval_manifold.inner(centroid, frechet, keepdim=True)
            alpha = (-k * ip).clamp_min(1.0 + eps)
            d = (torch.acosh(alpha) / torch.sqrt(k)).squeeze(-1)  # [BH, Q]

            # Spread: mean d(centroid[bh,q], v[bh,j]) over j → [BH, Q]
            cent_e = centroid.unsqueeze(2)  # [BH, Q, 1, D]
            v_e    = v_bh.unsqueeze(1)      # [BH, 1, N, D]
            ip_s   = eval_manifold.inner(cent_e, v_e, keepdim=True)
            alpha_s = (-k * ip_s).clamp_min(1.0 + eps)
            spread = (torch.acosh(alpha_s) / torch.sqrt(k)).squeeze(-1).mean(-1)  # [BH, Q]

            ratio = d / spread.clamp_min(1e-8)  # [BH, Q]

            layer_dists.extend(d.flatten().tolist())
            layer_ratios.extend(ratio.flatten().tolist())
            layer_entropies.extend(entropy.flatten().tolist())

        mean_d   = sum(layer_dists) / len(layer_dists)
        mean_r   = sum(layer_ratios) / len(layer_ratios)
        max_d    = max(layer_dists)
        max_r    = max(layer_ratios)
        mean_ent = sum(layer_entropies) / len(layer_entropies)

        results[layer_idx] = {
            "n_samples":    len(layer_dists),
            "mean_dist":    mean_d,
            "max_dist":     max_d,
            "mean_ratio":   mean_r,
            "max_ratio":    max_r,
            "mean_entropy": mean_ent,
        }

        print(f"  Layer {layer_idx:2d} | mean_entropy={mean_ent:.3f} | "
              f"mean_ratio={mean_r:.5f}  max_ratio={max_r:.5f} | "
              f"mean_dist={mean_d:.5f}  max_dist={max_d:.5f}")

        aggregate_dists.extend(layer_dists)
        aggregate_ratios.extend(layer_ratios)
        aggregate_entropies.extend(layer_entropies)

    # Pearson correlation between per-sample entropy and relative error
    import math
    n = len(aggregate_ratios)
    mean_r_all   = sum(aggregate_ratios)    / n
    mean_ent_all = sum(aggregate_entropies) / n
    cov  = sum((r - mean_r_all) * (e - mean_ent_all)
               for r, e in zip(aggregate_ratios, aggregate_entropies)) / n
    std_r   = math.sqrt(sum((r - mean_r_all)**2   for r in aggregate_ratios)    / n)
    std_ent = math.sqrt(sum((e - mean_ent_all)**2 for e in aggregate_entropies) / n)
    pearson_r = cov / (std_r * std_ent + 1e-12)

    print("\n── Aggregate ──")
    agg = {
        "n_samples":    n,
        "mean_dist":    sum(aggregate_dists)    / n,
        "max_dist":     max(aggregate_dists),
        "mean_ratio":   mean_r_all,
        "max_ratio":    max(aggregate_ratios),
        "mean_entropy": mean_ent_all,
        "pearson_r_entropy_vs_ratio": pearson_r,
    }
    print(f"  mean dist(centroid, Fréchet)        = {agg['mean_dist']:.6f}")
    print(f"  max  dist(centroid, Fréchet)        = {agg['max_dist']:.6f}")
    print(f"  mean dist / spread (relative error) = {agg['mean_ratio']:.6f}")
    print(f"  max  dist / spread                  = {agg['max_ratio']:.6f}")
    print(f"  mean attention entropy              = {agg['mean_entropy']:.4f}")
    print(f"  Pearson r(entropy, ratio)           = {pearson_r:.4f}")

    output = {
        "checkpoint":    str(args.checkpoint),
        "dataset":       cfg["dataset"],
        "val_acc":       ck["val_acc"],
        "curvature":     float(cfg["curvature"]),
        "n_batches":     args.n_batches,
        "batch_size":    args.batch_size,
        "sample_tokens": args.sample_tokens,
        "n_iter":        args.n_iter,
        "lr":            args.lr,
        "aggregate":     agg,
        "per_layer":     {str(k): v for k, v in results.items()},
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",    default="checkpoints_final/advanced/cifar100/hyperbolic_base/hyperbolic_cifar100.pt")
    parser.add_argument("--n_batches",    type=int,   default=5,    help="Number of validation batches")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--sample_tokens",type=int,   default=8,    help="Query tokens to sample per (image, head)")
    parser.add_argument("--n_iter",       type=int,   default=100,  help="Riemannian GD iterations for Fréchet mean")
    parser.add_argument("--lr",           type=float, default=0.3,  help="Riemannian GD step size")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--out",          default="experiments/hyperbolic/results_frechet.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
