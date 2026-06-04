#!/usr/bin/env python3
"""
Poincaré disk visualisation of hyperbolic ViT encoder (pooled) representations.

Uses geomstats TangentPCA at the Fréchet mean:
  https://geomstats.github.io/notebooks/06_practical_methods__riemannian_frechet_mean_and_tangent_pca.html

Install:  pip install geomstats

Usage:
    python plot_hyperbolic.py checkpoints/cifar100/hyperbolic_base/hyperbolic_cifar100.pt --output poincare.png --data-dir data/ --n-classes 20 --pts-per-class 50 --max-samples 2000
"""

import argparse
import sys
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def infer_config(state):
    # Classifier: support both old LorentzFullyConnected (.U) and new LorentzMLR (.z)
    if "classifier.U" in state:
        hidden_size = state["classifier.U"].shape[0]
        num_classes = state["classifier.U"].shape[1]
    elif "classifier.z" in state:
        # z shape: [num_classes+1, hidden_size]
        hidden_size = state["classifier.z"].shape[1]
        num_classes = state["classifier.z"].shape[0] - 1
    else:
        raise KeyError("Cannot infer classifier shape: no 'classifier.U' or 'classifier.z' key")

    # MLP dim: support both weight-norm (.v) and standard (.U)
    if "encoder.blocks.0.mlp.fc1.U" in state:
        mlp_dim = state["encoder.blocks.0.mlp.fc1.U"].shape[1]
    elif "encoder.blocks.0.mlp.fc1.v" in state:
        mlp_dim = state["encoder.blocks.0.mlp.fc1.v"].shape[1]
    else:
        mlp_dim = hidden_size * 4  # sensible default

    num_hidden_layers = max(
        int(k.split(".")[2]) for k in state if k.startswith("encoder.blocks.")
    ) + 1

    proj_v_key = "embedding.patch_embeddings.projection.v"
    if proj_v_key in state:
        patch_dim = state[proj_v_key].shape[0]
    else:
        patch_dim = state["embedding.patch_embeddings.projection.U"].shape[0]
    patch_size = int(round((patch_dim / 3) ** 0.5))

    num_patches = state["embedding.pos_space"].shape[1]
    image_size  = patch_size * int(round(num_patches ** 0.5))

    final_op = "cls" if "embedding.cls_token" in state else "pooling"

    head_keys = [
        k for k in state
        if k.startswith("encoder.blocks.0.attention.W_q.")
        and (k.endswith(".U") or k.endswith(".v"))
    ]
    use_weight_norm = "encoder.blocks.0.attention.W_q.0.v" in state

    return {
        "hidden_size":         hidden_size,
        "num_classes":         num_classes,
        "mlp_dim":             mlp_dim,
        "num_hidden_layers":   num_hidden_layers,
        "patch_size":          patch_size,
        "image_size":          image_size,
        "num_channels":        3,
        "num_attention_heads": len(head_keys),
        "final_op":            final_op,
        "use_weight_norm":     use_weight_norm,
    }


def build_model(state, cfg, curvature=1.0):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vit_hyperbolic.lorentz import Lorentz
    from vit_hyperbolic.model import LorentzViT

    model_cfg = {
        **cfg,
        "manifold":             Lorentz(curvature),
        "curvature":            curvature,
        "dropout_rate":         0.0,
        "attn_dropout_rate":    0.0,
        "concat_mode":          "direct",
        "compute_scores":       "lorentz_inner",
        "reset_params":         "lorentz_kaiming",
        "a_default":            0.1,
        "layernorm":            True,
        "stochastic_depth_rate": 0.0,
        "residual_weight_init": 0.0,
        "residual_delta_init":  2.0,
        "attn_mask":            None,
        "patch_embed_type":     "linear",
    }
    model = LorentzViT(model_cfg)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:3]}")
    model.eval()
    return model


def get_loader(num_classes, data_dir, batch_size=256):
    import torchvision
    import torchvision.transforms as T

    if num_classes == 100:
        mean = (0.50707516, 0.48654887, 0.44091784)
        std  = (0.26733429, 0.25643846, 0.27615047)
        DatasetCls = torchvision.datasets.CIFAR100
    elif num_classes == 10:
        mean = (0.49139968, 0.48215841, 0.44653091)
        std  = (0.24703223, 0.24348513, 0.26158784)
        DatasetCls = torchvision.datasets.CIFAR10
    else:
        raise ValueError(f"No known dataset for num_classes={num_classes}")

    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds = DatasetCls(data_dir, train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def extract_reps(model, loader, device, max_samples=None):
    reps, labels = [], []
    n = 0
    for images, y in loader:
        reps.append(model.encode(images.to(device)).cpu())
        labels.append(y)
        n += len(y)
        if max_samples is not None and n >= max_samples:
            break
    reps   = torch.cat(reps)
    labels = torch.cat(labels)
    if max_samples is not None:
        reps, labels = reps[:max_samples], labels[:max_samples]
    return reps, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output",        default="poincare.png")
    parser.add_argument("--data-dir",      default="data/")
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--n-classes",     type=int, default=20,
                        help="Number of classes to show (sampled randomly)")
    parser.add_argument("--pts-per-class", type=int, default=50)
    parser.add_argument("--max-samples",   type=int, default=2000,
                        help="Total representations to extract (speed vs coverage)")
    args = parser.parse_args()

    try:
        from geomstats.geometry.hyperboloid import Hyperboloid
        from geomstats.learning.pca import TangentPCA
    except ImportError:
        sys.exit("geomstats not found — install with:  pip install geomstats")

    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model"]
    print(f"Checkpoint: epoch={ckpt.get('epoch')}  acc={ckpt.get('acc')}")

    cfg = infer_config(state)
    print(f"Config: {cfg['num_classes']} classes  hidden={cfg['hidden_size']}  "
          f"layers={cfg['num_hidden_layers']}")

    model = build_model(state, cfg)

    # Extract classifier weight vectors [1+D, C] on CPU before moving model to GPU.
    # Supports both LorentzFullyConnected (.create_spacelike_vector) and LorentzMLR (.z/.a).
    clf = model.classifier
    if hasattr(clf, 'create_spacelike_vector'):
        V_full = clf.create_spacelike_vector().detach().cpu().numpy()   # [1+D, C]
    else:
        with torch.no_grad():
            sqrt_k = float(model.manifold.k().sqrt())
            a  = clf.a                                                   # [C]
            z  = clf.z                                                   # [C, D]
            nz = z.norm(dim=-1, keepdim=True)                           # [C, 1]
            wt = (torch.sinh(a / sqrt_k) * nz.squeeze(1)).unsqueeze(1) # [C, 1]
            ws = torch.cosh(a / sqrt_k).unsqueeze(1) * z               # [C, D]
            V_full = torch.cat([wt, ws], dim=1).T.cpu().numpy()        # [1+D, C]
    V_space = V_full[1:]  # [D, C]

    # Per-class a values for arc drawing
    if hasattr(clf, 'a'):
        a_vals = clf.a.detach().cpu().flatten().numpy()
        print(f"a_cls: mean={a_vals.mean():+.4f}  std={a_vals.std():.4f}  "
              f"min={a_vals.min():+.4f}  max={a_vals.max():+.4f}")
    else:
        a_vals = np.zeros(cfg["num_classes"])
    k_val = float(model.manifold.k().item())

    dev = torch.device("cuda" if args.device in ("gpu", "cuda") else args.device)
    model = model.to(dev)

    print("Loading data...")
    loader = get_loader(cfg["num_classes"], args.data_dir)

    print(f"Extracting representations (max {args.max_samples})...")
    reps, labels = extract_reps(model, loader, dev, max_samples=args.max_samples)
    # reps: [N, 1+D]  points on the Lorentz hyperboloid

    # ---- Normalise to safe Poincaré radius before TangentPCA ----------------
    # High-norm representations cause geomstats logmap to hit a numerical
    # singularity → NaN → SVD failure.  Normalising to TARGET_R preserves all
    # directional information (radial std < 1e-3, so every point is already at
    # the same radius).  After TangentPCA the 2D coordinates are rescaled back
    # to actual_r so the Poincaré disk plot shows points near the true edge.
    TARGET_R = 0.9
    r2    = TARGET_R ** 2
    t_new = (1.0 + r2) / (1.0 - r2)
    s_np  = reps[:, 1:].numpy()
    s_norm = np.linalg.norm(s_np, axis=1, keepdims=True).clip(1e-9)
    s_dir  = s_np / s_norm
    s_new  = s_dir * (2.0 * TARGET_R / (1.0 - r2))
    t_col  = np.full((len(reps), 1), t_new)
    reps_norm = np.concatenate([t_col, s_new], axis=1).astype(np.float64)
    # Per-point actual Poincaré radius (from original reps, before normalisation)
    r_actual = np.linalg.norm(s_np / (1.0 + reps[:, :1].numpy()), axis=1)  # [N]
    print(f"Original Poincaré radius: mean={r_actual.mean():.4f}  std={r_actual.std():.5f}  →  normalised to {TARGET_R} for TangentPCA")

    # ---- TangentPCA via geomstats ----------------------------------------
    print("Fitting TangentPCA (Fréchet mean + PCA on tangent space)...")
    space = Hyperboloid(dim=cfg["hidden_size"])
    tpca  = TangentPCA(space=space, n_components=2)
    coords_2d = tpca.fit_transform(reps_norm)  # [N, 2]

    var = tpca.explained_variance_ratio_
    print(f"Explained variance: PC1={var[0]:.2%}  PC2={var[1]:.2%}")

    # ---- 2D tangent coords → Poincaré disk ----------------------------------
    # Direction: unit vector from TangentPCA 2D coords (captures angular structure).
    # Radius: actual per-point Poincaré radius from the full-space representations.
    # This correctly shows all representations near the boundary while preserving
    # the angular clustering visible in the 2D projection.
    tangent_dir = coords_2d / np.linalg.norm(coords_2d, axis=1, keepdims=True).clip(1e-9)
    poincare    = r_actual[:, None] * tangent_dir  # [N, 2]

    print(f"Poincaré radii: mean={r_actual.mean():.4f}  std={r_actual.std():.5f}")

    # ---- Classifier weight vectors as Poincaré disk points ------------------
    # Project the full weight vector (time + spatial) onto the 2D PCA basis,
    # then map the 2D tangent coordinate to the Poincaré disk with tanh(‖v‖/2).
    V_2d    = tpca.components_ @ V_full     # [2, C]  (full projection incl. time)
    V_norm  = np.linalg.norm(V_2d, axis=0, keepdims=True).clip(1e-9)
    V_r     = np.tanh(V_norm / 2)
    V_poincare = V_r * V_2d / V_norm        # [2, C]

    V_2d_unit = V_2d / np.linalg.norm(V_2d, axis=0, keepdims=True).clip(1e-9)

    # ---- Select classes -----------------------------------------------------
    rng    = np.random.default_rng(42)
    shown  = sorted(rng.choice(cfg["num_classes"],
                               size=min(args.n_classes, cfg["num_classes"]),
                               replace=False).tolist())
    cmap   = plt.cm.gist_rainbow
    colors = cmap(np.linspace(0, 1, cfg["num_classes"]))

    # ---- MLR arc helper -----------------------------------------------------
    def draw_mlr_arc(target_ax, z_2d, a_val, k=1.0, **kwargs):
        """Draw the MLR decision boundary for direction z_2d and offset a_val.

        The boundary in the Poincaré disk is a circular arc whose closest point
        to the origin sits at Poincaré radius tanh(a*sqrt(k)/2).
        For a=0 the boundary is a diameter (passes through origin).
        """
        sqrt_k = np.sqrt(k)
        z_unit = np.array(z_2d, dtype=float)
        z_unit /= np.linalg.norm(z_unit) + 1e-12
        cos_r, sin_r = z_unit[0], z_unit[1]

        if abs(a_val) < 1e-4:
            perp = np.array([-z_unit[1], z_unit[0]])
            target_ax.plot([-perp[0], perp[0]], [-perp[1], perp[1]], **kwargs)
            return

        a_eff = abs(a_val) * sqrt_k
        try:
            cx_loc = np.cosh(a_eff) / np.sinh(a_eff)   # coth(a_eff)
            r_loc  = 1.0 / np.sinh(a_eff)               # csch(a_eff)
            theta_half = np.arccos(np.clip(1.0 / np.cosh(a_eff), -1, 1))
        except (ZeroDivisionError, FloatingPointError):
            return

        theta = np.linspace(np.pi - theta_half, np.pi + theta_half, 600)
        px_loc = cx_loc + r_loc * np.cos(theta)
        py_loc =          r_loc * np.sin(theta)

        # Rotate to actual z direction
        px = cos_r * px_loc - sin_r * py_loc
        py = sin_r * px_loc + cos_r * py_loc

        inside = px ** 2 + py ** 2 < 1.0
        if inside.sum() > 2:
            target_ax.plot(px[inside], py[inside], **kwargs)

    # ---- Plot ---------------------------------------------------------------
    fig, (ax, ax_ref) = plt.subplots(1, 2, figsize=(18, 9),
                                     gridspec_kw={"width_ratios": [1, 1]})

    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="black", linewidth=1.5, zorder=10))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.4, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.4, zorder=1)

    # Classifier decision boundaries (geodesic arcs shifted by a)
    for c in shown:
        a_c = float(a_vals[c]) if c < len(a_vals) else 0.0
        draw_mlr_arc(ax, V_2d_unit[:, c], a_c, k=k_val,
                     color=colors[c], linewidth=0.9, alpha=0.6, zorder=4)

    # Representations
    labs_np = labels.numpy()
    for c in shown:
        idx = np.where(labs_np == c)[0]
        if len(idx) > args.pts_per_class:
            idx = rng.choice(idx, size=args.pts_per_class, replace=False)
        ax.scatter(poincare[idx, 0], poincare[idx, 1],
                   color=colors[c], s=30, alpha=0.85, linewidths=0, zorder=5)

    # Classifier weight vectors as stars
    for c in shown:
        ax.scatter(V_poincare[0, c], V_poincare[1, c],
                   marker="*", color=colors[c], s=200,
                   edgecolors="black", linewidths=0.4, zorder=7)

    # Centroid class labels
    for c in shown:
        mask = labs_np == c
        if mask.sum() >= 3:
            cx, cy = poincare[mask, 0].mean(), poincare[mask, 1].mean()
            ax.text(cx, cy, str(c), fontsize=7, ha="center", va="center",
                    fontweight="bold", color="black", zorder=6)

    ax.set_title(
        f"Poincaré disk  ·  encoder (pooled) representations\n"
        f"TangentPCA direction  ·  PC1={var[0]:.1%}  PC2={var[1]:.1%}  ·  "
        f"actual radius={r_actual.mean():.4f} ± {r_actual.std():.5f}\n"
        f"geodesics = classifier boundaries (⟨x,V⟩_L=0)  ·  ★ = classifier weight  ·  "
        f"{args.n_classes}/{cfg['num_classes']} classes  ·  "
        f"{args.pts_per_class} pts/class",
        fontsize=10,
    )
    ax.axis("off")

    # ---- Reference panel: hyperplane offset vs a value ----------------------
    a_ref_vals = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
    ref_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(a_ref_vals)))

    ax_ref.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="black", linewidth=1.5, zorder=10))
    ax_ref.set_xlim(-1.1, 1.1)
    ax_ref.set_ylim(-1.1, 1.1)
    ax_ref.set_aspect("equal")
    ax_ref.axhline(0, color="lightgray", linewidth=0.4, zorder=1)
    ax_ref.axvline(0, color="lightgray", linewidth=0.4, zorder=1)

    # Draw arcs for z pointing right — all with the same direction so offsets
    # are purely due to a.  Each arc's foot sits at tanh(a*sqrt(k)/2) along x.
    z_ref = np.array([1.0, 0.0])
    for a_v, col in zip(a_ref_vals, ref_colors):
        draw_mlr_arc(ax_ref, z_ref, a_v, k=k_val,
                     color=col, linewidth=2.0, zorder=5)
        foot = np.tanh(a_v * np.sqrt(k_val) / 2.0)
        label = f"a={a_v:.2f}  (foot {foot:.3f})"
        ax_ref.scatter([foot], [0], color=col, s=60, zorder=6)
        # Stagger labels vertically to avoid overlap
        y_off = 0.06 * (a_ref_vals.index(a_v) - len(a_ref_vals) / 2)
        ax_ref.text(foot + 0.03, y_off, label, fontsize=8, color=col,
                    va="center", ha="left", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))

    # Mark origin
    ax_ref.scatter([0], [0], color="black", s=40, zorder=8)
    ax_ref.text(0.02, -0.06, "origin", fontsize=8, color="black")

    ax_ref.set_title(
        "MLR hyperplane offset vs a  (z = →, k=1)\n"
        "Arc = decision boundary  ·  dot = foot (closest point to origin)\n"
        "foot distance = tanh(a·√k / 2)  ·  a must reach ~0.5 for visible offset",
        fontsize=10,
    )
    ax_ref.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
