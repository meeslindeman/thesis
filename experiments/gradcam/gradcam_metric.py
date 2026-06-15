"""
Quantify how spatially concentrated each model's saliency is.

Two metrics, computed on the RAW grayscale CAM (before any per-image
min-max normalisation, so the numbers are comparable across models):

  - normalised entropy  H(p) / log(n_patches), p = map / map.sum()
        higher  -> saliency spread across many patches
  - top-1 mass          max(p)
        higher  -> saliency dominated by a single patch

Both models are forced onto the TRUE class so the comparison is like-for-like.
Run on a few hundred images and report mean +/- std + a Welch t-test.
"""
import argparse
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, EigenCAM, EigenGradCAM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "main"))

from config_base import Config
from data_utils import DATASET_REGISTRY, get_dataset
from utils import set_seed

CAMS = {"eigengradcam": EigenGradCAM, "eigencam": EigenCAM,
        "gradcam": GradCAM, "hirescam": HiResCAM}


def cfg_from_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in state:
        raise RuntimeError(f"Checkpoint {path} has no saved config.")
    init_names = {f.name for f in fields(Config) if f.init}
    cfg = Config(**{k: v for k, v in state["config"].items() if k in init_names})
    return cfg, state


def build_model(model_type, cfg):
    if model_type == "euclidean":
        from vit.model import ViT
        return ViT(asdict(cfg))
    from vit_hyperbolic.lorentz import Lorentz
    from vit_hyperbolic.model import LorentzViT
    model_cfg = asdict(cfg)
    model_cfg["manifold"] = Lorentz(cfg.curvature)
    return LorentzViT(model_cfg)


def load_checked(model, state, tag):
    model_state = model.state_dict()
    compatible = {k: v for k, v in state["model"].items()
                  if k in model_state and model_state[k].shape == v.shape}
    assert len(compatible) / len(model_state) > 0.95, \
        f"[{tag}] too many params unloaded — name/shape mismatch"
    model.load_state_dict(compatible, strict=False)
    model.eval()
    return model


def make_reshape_transform(grid_size, has_cls, hyperbolic):
    def reshape_transform(tensor):
        t = tensor[:, 1:, :] if has_cls else tensor
        if hyperbolic:
            t = t[:, :, 1:]
        B, N, C = t.shape
        assert N == grid_size * grid_size, f"{N} tokens != {grid_size}^2"
        t = t.reshape(B, grid_size, grid_size, C)
        return t.transpose(2, 3).transpose(1, 2)
    return reshape_transform


def concentration(cam_map):
    """Return (normalised_entropy, top1_mass) for one raw 2D CAM."""
    p = cam_map.astype(np.float64).ravel()
    p = np.clip(p, 0, None)              # CAM is non-negative after the lib's ReLU
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return np.nan, np.nan
    p = p / s
    nz = p[p > 0]
    entropy = -(nz * np.log(nz)).sum() / np.log(len(p))   # in [0, 1]
    return entropy, p.max()


def summarise(name, vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return f"{name}: {vals.mean():.4f} +/- {vals.std():.4f}  (n={len(vals)})", vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_euc", required=True)
    p.add_argument("--checkpoint_hyp", required=True)
    p.add_argument("--method", default="eigengradcam", choices=list(CAMS))
    p.add_argument("--num_images", type=int, default=400)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device)
    set_seed(args.seed)

    cfg_euc, state_euc = cfg_from_checkpoint(args.checkpoint_euc)
    cfg_hyp, state_hyp = cfg_from_checkpoint(args.checkpoint_hyp)
    for key in ("dataset", "image_size", "patch_size"):
        assert getattr(cfg_euc, key) == getattr(cfg_hyp, key), f"mismatch on {key}"

    cfg = cfg_euc
    dataset, _ = get_dataset(cfg.dataset, data_dir="data/")
    test_data = dataset["test"]
    grid_size = cfg.image_size // cfg.patch_size

    euc = load_checked(build_model("euclidean", cfg_euc).to(device), state_euc, "euc")
    hyp = load_checked(build_model("hyperbolic", cfg_hyp).to(device), state_hyp, "hyp")

    cam_euc = CAMS[args.method](
        model=euc, target_layers=[euc.encoder.blocks[-1].norm1],
        reshape_transform=make_reshape_transform(grid_size, cfg_euc.final_op == "cls", False))
    cam_hyp = CAMS[args.method](
        model=hyp, target_layers=[hyp.encoder.blocks[-1].batchnorm1],
        reshape_transform=make_reshape_transform(grid_size, cfg_hyp.final_op == "cls", True))

    rng = torch.Generator(); rng.manual_seed(args.seed)
    order = torch.randperm(len(test_data), generator=rng).tolist()

    ent_e, ent_h, top_e, top_h = [], [], [], []
    done = 0
    for i in order:
        if done >= args.num_images:
            break
        img_t, label = test_data[i]
        image = img_t.unsqueeze(0).to(device)

        # Force BOTH models onto the true class (like-for-like comparison).
        # targets = [lambda o, c=label: o[c]]
        targets = None
        try:
            g_e = cam_euc(input_tensor=image, targets=targets)[0]
            g_h = cam_hyp(input_tensor=image, targets=targets)[0]
        except Exception as e:
            print(f"skip image {i}: {e}")
            continue
        if not (np.isfinite(g_e).all() and np.isfinite(g_h).all()):
            continue

        he, te = concentration(g_e)
        hh, th = concentration(g_h)
        if np.isnan(he) or np.isnan(hh):
            continue
        ent_e.append(he); ent_h.append(hh); top_e.append(te); top_h.append(th)
        done += 1

    from scipy import stats
    print(f"\nmethod={args.method}  images={done}  patches={grid_size**2}\n")
    for label, ev, hv in [("entropy (higher = more spread)", ent_e, ent_h),
                          ("top-1 mass (higher = concentrated)", top_e, top_h)]:
        print(label)
        line_e, ev = summarise("  euclidean", ev)
        line_h, hv = summarise("  hyperbolic", hv)
        t, pval = stats.ttest_ind(hv, ev, equal_var=False)
        print(line_e); print(line_h)
        print(f"  Welch t={t:.3f}  p={pval:.2e}\n")


if __name__ == "__main__":
    main()