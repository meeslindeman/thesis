"""
Substitute the exact Fréchet mean at every value-aggregation step during validation
and compare accuracy against the centroid baseline.

Three honest outcomes:
  Δ ≈ 0  → centroid and Fréchet are interchangeable on learned representations
  Δ < 0  → model adapted to the centroid it trained with; still supports keeping it
  Δ > 0  → Fréchet strictly better; the result that would undercut the centroid choice

Usage:
    python frechet_eval.py [--checkpoint PATH] [--n_iter 50] [--lr 0.3] [--device cuda]
"""

import sys
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "main"))


# ── Fréchet mean (fully batched, reused from frechet_divergence.py) ──────────

def frechet_means_batched(manifold, points_bhnd, weights_bhqn, n_iter=50, lr=0.3):
    """
    points_bhnd  : [BH, N, D]
    weights_bhqn : [BH, Q, N]   — for full aggregation Q == N
    returns      : [BH, Q, D]
    """
    k   = manifold.k().to(points_bhnd)
    eps = manifold.eps
    sqrt_k = torch.sqrt(k)

    with torch.no_grad():
        # Inline centroid to avoid calling the (potentially patched) lorentz_centroid
        num      = torch.einsum("...qk,...kd->...qd", weights_bhqn, points_bhnd)
        t_num    = num[..., :1]
        s_num    = num[..., 1:]
        denom_sq = (t_num * t_num - (s_num * s_num).sum(-1, keepdim=True)).clamp_min(eps)
        mu       = (num / (torch.sqrt(denom_sq) * sqrt_k)).clone()

        for _ in range(n_iter):
            mu_e  = mu.unsqueeze(2)            # [BH, Q, 1, D]
            pts_e = points_bhnd.unsqueeze(1)   # [BH, 1, N, D]

            xy = (-mu_e[..., :1] * pts_e[..., :1]
                  + (mu_e[..., 1:] * pts_e[..., 1:]).sum(-1, keepdim=True))

            alpha  = (-k * xy).clamp_min(1.0 + eps)
            ac     = torch.acosh(alpha)
            denom  = torch.sqrt((alpha * alpha - 1).clamp_min(0.0))
            factor = torch.where(denom < eps, torch.ones_like(denom), ac / denom)

            u    = pts_e + (k * xy) * mu_e    # [BH, Q, N, D]
            logs = factor * u

            grad = -2.0 * (weights_bhqn.unsqueeze(-1) * logs).sum(2)  # [BH, Q, D]

            step   = -lr * grad
            vv     = (-step[..., :1] * step[..., :1]
                      + (step[..., 1:] * step[..., 1:]).sum(-1, keepdim=True)).clamp_min(1e-8)
            norm_v = torch.sqrt(vv)
            arg    = (sqrt_k * norm_v).clamp(max=85.0)
            left   = torch.cosh(arg)
            right  = torch.where(arg < 1e-6, torch.ones_like(arg), torch.sinh(arg) / arg)
            mu_new = left * mu + right * step

            space = mu_new[..., 1:]
            time  = torch.sqrt((1.0 / k + (space * space).sum(-1, keepdim=True)).clamp_min(eps))
            mu    = torch.cat([time, space], dim=-1)

    return mu  # [BH, Q, D]


# ── Validation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device)

    ck  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model      : {ck['model_name']}  val_acc={ck['val_acc']:.4f}  dataset={cfg['dataset']}")

    from vit_hyperbolic.lorentz import Lorentz
    from vit_hyperbolic.model   import LorentzViT
    from data_utils import get_dataset, get_dataloader

    manifold = Lorentz(cfg["curvature"])
    cfg["manifold"] = manifold
    model = LorentzViT(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    datasets, _ = get_dataset(cfg["dataset"], augmentation="basic")
    loaders = get_dataloader(datasets, batch_size=args.batch_size)
    val_loader = loaders["validation"]

    n_val = len(datasets["validation"])
    print(f"Validation : {n_val} samples  batch={args.batch_size}")

    # ── Pass 1: centroid baseline ────────────────────────────────────────────
    print("\n[1/2] Centroid baseline …")
    acc_centroid = evaluate(model, val_loader, device)
    print(f"  centroid  acc = {acc_centroid:.4f}  ({acc_centroid*100:.2f}%)")

    # ── Patch all attention manifolds to use Fréchet mean ────────────────────
    originals = {}
    for layer_idx, block in enumerate(model.encoder.blocks):
        attn_manifold = block.attention.manifold
        orig = attn_manifold.lorentz_centroid
        originals[layer_idx] = orig

        def _make_frechet(orig_fn, mfld):
            def frechet_aggregation(xs, w):
                # xs: [B, H, N, D_head], w: [B, H, N, N]
                B, H, N, D = xs.shape
                v_bh = xs.reshape(B * H, N, D)
                a_bh = w.reshape(B * H, N, N)   # [BH, Q=N, K=N]
                fm   = frechet_means_batched(mfld, v_bh, a_bh,
                                             n_iter=args.n_iter, lr=args.lr)
                return fm.reshape(B, H, N, D)
            return frechet_aggregation

        attn_manifold.lorentz_centroid = _make_frechet(orig, attn_manifold)

    # ── Pass 2: Fréchet mean ─────────────────────────────────────────────────
    print(f"\n[2/2] Fréchet mean  (n_iter={args.n_iter}, lr={args.lr}) …")
    acc_frechet = evaluate(model, val_loader, device)
    print(f"  fréchet   acc = {acc_frechet:.4f}  ({acc_frechet*100:.2f}%)")

    # Restore
    for layer_idx, block in enumerate(model.encoder.blocks):
        block.attention.manifold.lorentz_centroid = originals[layer_idx]

    # ── Result ───────────────────────────────────────────────────────────────
    delta = acc_frechet - acc_centroid
    print(f"\n── Result ──")
    print(f"  Δ acc = {delta:+.4f}  ({delta*100:+.2f} pp)")
    if abs(delta) < 0.002:
        verdict = "centroid and Fréchet mean are interchangeable on these representations"
    elif delta < 0:
        verdict = "model adapted to centroid (Fréchet substitution hurts slightly); supports keeping centroid"
    else:
        verdict = "Fréchet strictly better — centroid is a non-trivial approximation"
    print(f"  Verdict : {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints_final/advanced/cifar100/hyperbolic_base/hyperbolic_cifar100.pt")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--n_iter",     type=int,   default=50,  help="Fréchet GD iterations (50 is sufficient given centroid warm-start)")
    parser.add_argument("--lr",         type=float, default=0.3)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
