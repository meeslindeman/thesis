#!/usr/bin/env python3
"""Clean Poincaré disk plot for LViT (vit_hypercore) checkpoints."""

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from pathlib import Path

plt.rcParams.update({
    "font.size":        15,
    "axes.linewidth":   0.8,
})

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.228, 0.224, 0.225)

NORM_STATS = {
    "cifar10":        {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "cifar100":       {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)},
    "tiny_imagenet":  {"mean": (0.4802, 0.4481, 0.3975),             "std": (0.2302, 0.2265, 0.2262)},
}


# ── model helpers ────────────────────────────────────────────────────────────

def extract_config(ckpt):
    cfg = ckpt["config"]
    return {
        "hidden_size":         cfg["hidden_size"],
        "num_classes":         cfg["num_classes"],
        "num_hidden_layers":   cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "mlp_dim":             cfg["mlp_dim"],
        "patch_size":          cfg["patch_size"],
        "image_size":          cfg.get("image_size", 32),
        "num_channels":        cfg.get("num_channels", 3),
        "curvature":           cfg.get("curvature", 1.0),
        "dropout_rate":        cfg.get("dropout_rate", 0.1),
        "dataset":             cfg.get("dataset"),
    }


def build_model(state, cfg):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from vit_hypercore.manifold.lorentz import Lorentz
    from vit_hypercore.model import LViT

    c = cfg["curvature"]
    hidden_channel = cfg["hidden_size"] // cfg["num_attention_heads"]

    model = LViT(
        manifold_in      = Lorentz(c),
        manifold_hidden  = Lorentz(c),
        manifold_out     = Lorentz(c),
        image_size       = cfg["image_size"],
        patch_size       = cfg["patch_size"],
        num_layers       = cfg["num_hidden_layers"],
        in_channel       = cfg["num_channels"],
        hidden_channel   = hidden_channel,
        out_channel      = cfg["num_classes"],
        mlp_hidden_size  = cfg["mlp_dim"] + 1,
        num_heads        = cfg["num_attention_heads"],
        dropout          = cfg["dropout_rate"],
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_loader(dataset: str, data_dir: str, batch_size: int = 256):
    import torchvision
    import torchvision.transforms as T

    stats = NORM_STATS.get(dataset, {"mean": _IMAGENET_MEAN, "std": _IMAGENET_STD})
    tf    = T.Compose([T.ToTensor(), T.Normalize(stats["mean"], stats["std"])])

    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    elif dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    elif dataset == "tiny_imagenet":
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(repo_root, "experiments", "main"))
        from data_utils import TinyImageNet
        ds = TinyImageNet(data_dir, train=False, transform=tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)


@torch.no_grad()
def extract_reps(model, loader, device, max_samples=None):
    # Setting out_channel=0 makes LViT return the lorentzian centroid directly.
    orig = model.out_channel
    model.out_channel = 0
    reps, labels = [], []
    n = 0
    for images, y in loader:
        reps.append(model(images.to(device)).cpu())
        labels.append(y)
        n += len(y)
        if max_samples and n >= max_samples:
            break
    model.out_channel = orig
    reps   = torch.cat(reps)
    labels = torch.cat(labels)
    if max_samples:
        reps, labels = reps[:max_samples], labels[:max_samples]
    return reps, labels


# ── drawing ──────────────────────────────────────────────────────────────────

def draw_mlr_arc(ax, z_2d, a_val, k=1.0, **kwargs):
    sqrt_k = np.sqrt(k)
    z_unit = np.array(z_2d, dtype=float)
    z_unit /= np.linalg.norm(z_unit) + 1e-12
    cos_r, sin_r = z_unit[0], z_unit[1]

    if abs(a_val) < 1e-4:
        perp = np.array([-z_unit[1], z_unit[0]])
        ax.plot([-perp[0], perp[0]], [-perp[1], perp[1]], **kwargs)
        return

    a_eff = abs(a_val) * sqrt_k
    try:
        cx_loc     = np.cosh(a_eff) / np.sinh(a_eff)
        r_loc      = 1.0 / np.sinh(a_eff)
        theta_half = np.arccos(np.clip(1.0 / np.cosh(a_eff), -1, 1))
    except (ZeroDivisionError, FloatingPointError):
        return

    theta  = np.linspace(np.pi - theta_half, np.pi + theta_half, 600)
    px_loc = cx_loc + r_loc * np.cos(theta)
    py_loc =          r_loc * np.sin(theta)
    px = cos_r * px_loc - sin_r * py_loc
    py = sin_r * px_loc + cos_r * py_loc

    inside = px ** 2 + py ** 2 < 1.0
    if inside.sum() > 2:
        ax.plot(px[inside], py[inside], **kwargs)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset",        default=None,
                        help="Dataset name (default: read from checkpoint config)")
    parser.add_argument("--data-dir",       default="data/")
    parser.add_argument("--output",         default=None)
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--n-classes",      type=int, default=None)
    parser.add_argument("--pts-per-class",  type=int, default=50)
    parser.add_argument("--max-samples",    type=int, default=2000)
    parser.add_argument("--no-classifiers", dest="classifiers", action="store_false")
    parser.add_argument("--tangent",        action="store_true")
    args = parser.parse_args()

    try:
        from geomstats.geometry.hyperboloid import Hyperboloid
        from geomstats.learning.pca import TangentPCA
    except ImportError:
        sys.exit("geomstats not found — install with:  pip install geomstats")

    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model"]
    cfg   = extract_config(ckpt)
    num_classes = cfg["num_classes"]
    n_show      = args.n_classes or num_classes

    dataset = args.dataset or cfg.get("dataset")
    if dataset is None:
        sys.exit("Could not determine dataset — pass --dataset or ensure the checkpoint has a 'config.dataset' field.")

    print(f"Checkpoint: epoch={ckpt.get('epoch')}  acc={ckpt.get('val_acc')}  dataset={dataset}")

    model = build_model(state, cfg)
    dev   = torch.device("cuda" if args.device in ("gpu", "cuda") else args.device)
    model = model.to(dev)

    print("Loading data...")
    loader = get_loader(dataset, args.data_dir)

    effective_max = max(args.max_samples, args.pts_per_class * num_classes)
    print(f"Extracting representations (max {effective_max})...")
    reps, labels = extract_reps(model, loader, dev, max_samples=effective_max)

    # Poincaré radius from Lorentz coordinates
    s_np     = reps[:, 1:].numpy()
    r_actual = np.linalg.norm(s_np / (1.0 + reps[:, :1].numpy()), axis=1)

    print(f"Mean radius: {r_actual.mean():.4f}  Std: {r_actual.std():.4f}"
          f"  (min={r_actual.min():.4f}  max={r_actual.max():.4f})")

    CLIP_R  = 0.97
    reps_np = reps.numpy().astype(np.float64)
    if r_actual.max() > CLIP_R:
        r2   = CLIP_R ** 2
        t_c  = (1.0 + r2) / (1.0 - r2)
        s_c  = s_np / np.linalg.norm(s_np, axis=1, keepdims=True).clip(1e-9) * (2.0 * CLIP_R / (1.0 - r2))
        mask = r_actual > CLIP_R
        reps_np[mask, 0]  = t_c
        reps_np[mask, 1:] = s_c[mask].astype(np.float64)

    print("Fitting TangentPCA...")
    hyp_dim  = reps.shape[1] - 1   # spatial dims = total - 1 (time coord)
    space    = Hyperboloid(dim=hyp_dim)
    tpca     = TangentPCA(space=space, n_components=2)
    coords2d = tpca.fit_transform(reps_np)

    tangent_dir = coords2d / np.linalg.norm(coords2d, axis=1, keepdims=True).clip(1e-9)
    poincare    = r_actual[:, None] * tangent_dir

    # Classifier weight vectors
    if args.classifiers:
        clf = model.classifier
        with torch.no_grad():
            c_val  = model.manifold_out.c
            sqrt_k = float(c_val.sqrt() if hasattr(c_val, "sqrt") else c_val ** 0.5)
            a  = clf.a
            z  = clf.z
            nz = z.norm(dim=-1, keepdim=True)
            wt = (torch.sinh(a / sqrt_k) * nz.squeeze(1)).unsqueeze(1)
            ws = torch.cosh(a / sqrt_k).unsqueeze(1) * z
            V_full = torch.cat([wt, ws], dim=1).T.cpu().numpy()

        a_vals    = clf.a.detach().cpu().flatten().numpy()
        k_val     = float(c_val)
        V_2d      = tpca.components_ @ V_full
        V_2d_unit = V_2d / np.linalg.norm(V_2d, axis=0, keepdims=True).clip(1e-9)

    # Select classes
    rng        = np.random.default_rng(42)
    labs       = labels.numpy()
    shown      = sorted(rng.choice(num_classes, size=min(n_show, num_classes), replace=False).tolist())
    cmap       = plt.cm.viridis
    color_norm = Normalize(vmin=0, vmax=len(shown) - 1)
    color_idx  = {c: i for i, c in enumerate(shown)}
    def cc(c): return cmap(color_norm(color_idx[c]))

    # ── plot ─────────────────────────────────────────────────────────────────
    if args.tangent:
        fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    def add_colorbar(target_ax):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=target_ax, fraction=0.04, pad=0.02)
        cbar.set_label("Class")
        cbar.ax.yaxis.set_tick_params(length=2)
        if len(shown) <= 20:
            cbar.set_ticks(range(len(shown)))
            cbar.set_ticklabels([str(c) for c in shown])

    # ── Poincaré disk ────────────────────────────────────────────────────────
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="black", linewidth=1.0, zorder=10))

    if args.classifiers:
        for c in shown:
            a_c = float(a_vals[c]) if c < len(a_vals) else 0.0
            draw_mlr_arc(ax, V_2d_unit[:, c], a_c, k=k_val,
                         color=cc(c), linewidth=0.8, alpha=0.5, zorder=4)

    for c in shown:
        idx = np.where(labs == c)[0]
        if len(idx) > args.pts_per_class:
            idx = rng.choice(idx, size=args.pts_per_class, replace=False)
        ax.scatter(poincare[idx, 0], poincare[idx, 1],
                   s=6, alpha=0.75, color=cc(c), linewidths=0, zorder=5)

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("$z_0$")
    ax.set_ylabel("$z_1$")
    ax.tick_params(direction="in", length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    add_colorbar(ax)

    # ── Tangent space panel ──────────────────────────────────────────────────
    if args.tangent:
        for c in shown:
            idx = np.where(labs == c)[0]
            if len(idx) > args.pts_per_class:
                idx = rng.choice(idx, size=args.pts_per_class, replace=False)
            ax_t.scatter(coords2d[idx, 0], coords2d[idx, 1],
                         s=6, alpha=0.75, color=cc(c), linewidths=0)

        lim = np.abs(coords2d[np.isin(labs, shown)]).max() * 1.05
        ax_t.set_xlim(-lim, lim)
        ax_t.set_ylim(-lim, lim)
        ax_t.set_aspect("equal", adjustable="box")
        ax_t.set_xlabel("$z_0$")
        ax_t.set_ylabel("$z_1$")
        ax_t.tick_params(direction="in", length=3)
        for spine in ax_t.spines.values():
            spine.set_visible(True)
        add_colorbar(ax_t)

    fig.tight_layout()

    plots_dir = Path("plotting/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = args.output or str(plots_dir / f"poincare_hypercore_{dataset}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
