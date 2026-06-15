import argparse
import sys
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "main"))

from config_base import Config
from data_utils import DATASET_REGISTRY, get_dataset
from utils import set_seed

CAMS = {"eigengradcam": EigenGradCAM, "eigencam": EigenCAM,
        "gradcam": GradCAM, "hirescam": HiResCAM}

# Buckets, in the order we want rows to appear in the figure.
BUCKETS = ["hyp_only", "euc_only", "both_wrong", "both_right"]
BUCKET_LABEL = {
    "both_right": "both correct",
    "both_wrong": "both wrong",
    "euc_only":   "euclidean correct, hyperbolic wrong",
    "hyp_only":   "hyperbolic correct, euclidean wrong",
}


def cfg_from_checkpoint(path: str):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in state:
        raise RuntimeError(f"Checkpoint {path} has no saved config.")
    init_names = {f.name for f in fields(Config) if f.init}
    cfg = Config(**{k: v for k, v in state["config"].items() if k in init_names})
    return cfg, state


def build_model(model_type: str, cfg):
    if model_type == "euclidean":
        from vit.model import ViT
        return ViT(asdict(cfg))
    from vit_hyperbolic.lorentz import Lorentz
    from vit_hyperbolic.model import LorentzViT
    model_cfg = asdict(cfg)
    model_cfg["manifold"] = Lorentz(cfg.curvature)
    return LorentzViT(model_cfg)


def load_checked(model, state, tag: str):
    """Load weights, but refuse to silently run a half-initialised model."""
    model_state = model.state_dict()
    compatible = {k: v for k, v in state["model"].items()
                  if k in model_state and model_state[k].shape == v.shape}
    missing = sorted(set(model_state) - set(compatible))
    print(f"[{tag}] loaded {len(compatible)}/{len(model_state)} params; "
          f"first missing: {missing[:8]}")
    assert len(compatible) / len(model_state) > 0.95, \
        f"[{tag}] too many params unloaded ({len(missing)}) — name/shape mismatch"
    model.load_state_dict(compatible, strict=False)
    model.eval()
    return model


def make_reshape_transform(grid_size: int, has_cls: bool, hyperbolic: bool):
    def reshape_transform(tensor):
        # tensor: [B, N(+1), C]  (euclidean) or [B, N(+1), C+1] (Lorentz points)
        t = tensor[:, 1:, :] if has_cls else tensor
        if hyperbolic:
            t = t[:, :, 1:]                     # drop time component -> space coords
        B, N, C = t.shape
        assert N == grid_size * grid_size, \
            f"{N} tokens != {grid_size}^2 — CLS / token-count mismatch"
        t = t.reshape(B, grid_size, grid_size, C)
        return t.transpose(2, 3).transpose(1, 2)   # [B, C, H, W]
    return reshape_transform


def denormalize(tensor, mean, std) -> np.ndarray:
    t = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)


def upsample_img(img: np.ndarray, size: int = 128) -> np.ndarray:
    import torch.nn.functional as F
    t = torch.from_numpy(img).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze().permute(1, 2, 0).numpy()


def run_cam(cam, image, target_class, tag):
    """Return a grayscale CAM, guarding against NaNs from the manifold path."""
    targets = ([lambda o, c=target_class: o[c]]
               if target_class is not None else None)
    try:
        g = cam(input_tensor=image, targets=targets)[0]
    except Exception as e:        # numerical blow-ups in midpoint pooling, etc.
        print(f"[{tag}] CAM failed: {e}")
        return None
    if not np.isfinite(g).all():
        print(f"[{tag}] CAM produced non-finite values — zeroing them")
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_euc", required=True)
    p.add_argument("--checkpoint_hyp", required=True)
    p.add_argument("--method", default="eigengradcam", choices=list(CAMS))
    p.add_argument("--num_images", type=int, default=8,
                   help="total images to collect across all buckets")
    p.add_argument("--per_bucket", type=int, default=0,
                   help="if >0, cap images per bucket (useful to surface disagreements)")
    p.add_argument("--target_class", type=int, default=None,
                   help="explain this class; default: each model's own prediction")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="experiments/gradcam/out")
    args = p.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    set_seed(args.seed)

    cfg_euc, state_euc = cfg_from_checkpoint(args.checkpoint_euc)
    cfg_hyp, state_hyp = cfg_from_checkpoint(args.checkpoint_hyp)

    # The comparison is only meaningful if the data pipeline matches.
    for key in ("dataset", "image_size", "patch_size"):
        assert getattr(cfg_euc, key) == getattr(cfg_hyp, key), \
            f"checkpoints disagree on {key}: {getattr(cfg_euc, key)} vs {getattr(cfg_hyp, key)}"
    print(f"euc final_op: {cfg_euc.final_op} | hyp final_op: {cfg_hyp.final_op}")
    if cfg_euc.final_op != cfg_hyp.final_op:
        print("WARNING: different pooling -> different gradient structure; "
              "maps are not strictly comparable.")

    cfg = cfg_euc
    dataset, _ = get_dataset(cfg.dataset, data_dir="data/")
    test_data = dataset["test"]
    meta = DATASET_REGISTRY[cfg.dataset]
    mean, std = meta["mean"], meta["std"]
    class_names = getattr(test_data, "classes", None)
    grid_size = cfg.image_size // cfg.patch_size

    euc = load_checked(build_model("euclidean", cfg_euc).to(device), state_euc, "euc")
    hyp = load_checked(build_model("hyperbolic", cfg_hyp).to(device), state_hyp, "hyp")

    cam_euc = CAMS[args.method](
        model=euc, target_layers=[euc.encoder.blocks[-1].norm1],   # not [...blocks[-1]]
        reshape_transform=make_reshape_transform(
            grid_size, cfg_euc.final_op == "cls", hyperbolic=False),
    )
    cam_hyp = CAMS[args.method](
        model=hyp, target_layers=[hyp.encoder.blocks[-1].batchnorm1],   # was [...blocks[-1]]
        reshape_transform=make_reshape_transform(
            grid_size, cfg_hyp.final_op == "cls", hyperbolic=True),
    )

    rng = torch.Generator(); rng.manual_seed(args.seed)
    order = torch.randperm(len(test_data), generator=rng).tolist()

    samples = []
    bucket_counts = {b: 0 for b in BUCKETS}
    for i in order:
        if len(samples) >= args.num_images:
            break
        img_t, label = test_data[i]
        image = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            pe = euc(image).argmax(dim=1).item()
            ph = hyp(image).argmax(dim=1).item()
        ce, ch = pe == label, ph == label
        bucket = ("both_right" if ce and ch else
                  "both_wrong" if not ce and not ch else
                  "euc_only" if ce else "hyp_only")
        if args.per_bucket and bucket_counts[bucket] >= args.per_bucket:
            continue

        g_euc = run_cam(cam_euc, image, args.target_class, "euc")
        g_hyp = run_cam(cam_hyp, image, args.target_class, "hyp")
        if g_euc is None or g_hyp is None:
            continue

        print(f"euc cam: max={g_euc.max():.3g} std={g_euc.std():.3g} | "
        f"hyp cam: max={g_hyp.max():.3g} std={g_hyp.std():.3g}")

        img_np = denormalize(img_t, mean, std)
        samples.append(dict(
            img=img_np,
            ov_euc=show_cam_on_image(img_np, g_euc, use_rgb=True).astype(np.float32) / 255.0,
            ov_hyp=show_cam_on_image(img_np, g_hyp, use_rgb=True).astype(np.float32) / 255.0,
            label=label, pe=pe, ph=ph, bucket=bucket,
        ))
        bucket_counts[bucket] += 1

    samples.sort(key=lambda s: BUCKETS.index(s["bucket"]))
    print("bucket counts:", bucket_counts)

    # ── Grid: one row per image, 3 cols = original | euc overlay | hyp overlay
    disp = 128
    n = max(len(samples), 1)
    fig, axes = plt.subplots(n, 3, figsize=(3 * 3.0, n * 3.2))
    axes = np.array(axes).reshape(n, 3)

    def name(idx):
        return class_names[idx] if class_names else str(idx)

    for r, s in enumerate(samples):
        tick_e = "✓" if s["pe"] == s["label"] else "✗"
        tick_h = "✓" if s["ph"] == s["label"] else "✗"

        axes[r, 0].imshow(upsample_img(s["img"], disp), interpolation="nearest")
        axes[r, 0].set_title(name(s["label"]), fontsize=10)

        axes[r, 1].imshow(upsample_img(s["ov_euc"], disp), interpolation="nearest")
        axes[r, 1].set_title(f"{name(s['pe'])} {tick_e}", fontsize=10)

        axes[r, 2].imshow(upsample_img(s["ov_hyp"], disp), interpolation="nearest")
        axes[r, 2].set_title(f"{name(s['ph'])} {tick_h}", fontsize=10)

        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    fig.suptitle(f"{args.method.upper()} · euclidean vs hyperbolic ViT · {cfg.dataset}",
                 fontsize=12, y=1.005)
    plt.tight_layout()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gradcam_compare_{args.method}_{cfg.dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}  ({len(samples)} images)")


if __name__ == "__main__":
    main()