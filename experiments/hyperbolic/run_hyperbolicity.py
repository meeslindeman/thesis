"""
Gromov δ-hyperbolicity on CIFAR-10 in three modes:

  vgg         -- VGG-16 ImageNet features (the paper's approach, Euclidean distances)
  euclidean   -- Our trained Euclidean ViT encode() embeddings (Euclidean distances)
  hyperbolic  -- Our trained Lorentz ViT encode() embeddings (Lorentz geodesic distances)

The relative hyperbolicity δ_rel = 2δ / diam is reported.
Lower δ_rel means the feature space is more tree-like (hyperbolic).

Usage:
    python run_hyperbolicity.py --mode vgg
    python run_hyperbolicity.py --mode euclidean
    python run_hyperbolicity.py --mode hyperbolic
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from scipy.spatial import distance_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperbolicity import delta_hyp

DATA_DIR = ROOT / "data"

DEFAULT_CKPT = {
    "euclidean":  "checkpoints_final/advanced/cifar10/euclidean_base/euclidean_cifar10.pt",
    "hyperbolic": "checkpoints_final/advanced/cifar10/hyperbolic_base/hyperbolic_cifar10.pt",
}


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_vgg_features(n_samples: int, batch_size: int, device: torch.device) -> np.ndarray:
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    # Mirror the original paper: features → flatten → classifier[:-1] (4096-d)
    model = nn.Sequential(
        vgg.features,
        nn.Flatten(),
        *list(vgg.classifier.children())[:-1],
    ).to(device).eval()

    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="VGG features"):
            feats.append(model(imgs.to(device)).cpu().numpy())
            if sum(len(f) for f in feats) >= n_samples:
                break
    return np.concatenate(feats)[:n_samples]


def extract_euclidean_features(checkpoint: str, n_samples: int, batch_size: int, device: torch.device) -> np.ndarray:
    from vit.model import ViT
    from data_utils import get_dataset, get_dataloader

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    print(f"  val_acc={ck['val_acc']:.4f}  dataset={cfg['dataset']}")

    model = ViT(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    datasets, _ = get_dataset("cifar10", augmentation="basic")
    loader = get_dataloader(datasets, batch_size=batch_size)["test"]

    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Euclidean ViT features"):
            feats.append(model.encode(imgs.to(device)).cpu().numpy())
            if sum(len(f) for f in feats) >= n_samples:
                break
    return np.concatenate(feats)[:n_samples]


def extract_lorentz_distances(checkpoint: str, n_samples: int, batch_size: int, device: torch.device) -> np.ndarray:
    """Returns an (N, N) pairwise Lorentz geodesic distance matrix."""
    from vit_hyperbolic.lorentz import Lorentz
    from vit_hyperbolic.model import LorentzViT
    from data_utils import get_dataset, get_dataloader

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    print(f"  val_acc={ck['val_acc']:.4f}  dataset={cfg['dataset']}  curvature={cfg['curvature']}")

    manifold = Lorentz(cfg["curvature"])
    cfg["manifold"] = manifold
    model = LorentzViT(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    datasets, _ = get_dataset("cifar10", augmentation="basic")
    loader = get_dataloader(datasets, batch_size=batch_size)["test"]

    feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Lorentz ViT features"):
            feats.append(model.encode(imgs.to(device)).cpu())
            if sum(len(f) for f in feats) >= n_samples:
                break

    X = torch.cat(feats)[:n_samples].to(device)  # [N, D]
    N = len(X)
    k   = manifold.k().to(X)
    eps = manifold.eps

    print(f"  Computing {N}×{N} Lorentz geodesic distance matrix ...")
    chunk = 256
    dists = np.zeros((N, N), dtype=np.float32)
    for i in range(0, N, chunk):
        xi = X[i:i + chunk]
        for j in range(0, N, chunk):
            xj = X[j:j + chunk]
            # Lorentz inner product: -t_i * t_j + s_i · s_j
            ip = -xi[:, :1] @ xj[:, :1].T + xi[:, 1:] @ xj[:, 1:].T
            alpha = (-k * ip).clamp_min(1.0 + eps)
            d = (torch.acosh(alpha) / torch.sqrt(k)).cpu().numpy()
            dists[i:i + chunk, j:j + chunk] = d

    return dists


# ── Delta computation ─────────────────────────────────────────────────────────

def batched_delta(distmat: np.ndarray, n_tries: int, sample_size: int) -> tuple[float, float]:
    N = len(distmat)
    sample_size = min(sample_size, N)
    vals = []
    for _ in tqdm(range(n_tries), desc="δ trials"):
        idx = np.random.choice(N, sample_size, replace=False)
        sub = distmat[np.ix_(idx, idx)]
        diam = np.max(sub)
        if diam < 1e-8:
            continue
        vals.append(2.0 * delta_hyp(sub) / diam)
    return float(np.mean(vals)), float(np.std(vals))


def batched_delta_from_features(X: np.ndarray, n_tries: int, sample_size: int) -> tuple[float, float]:
    N = len(X)
    sample_size = min(sample_size, N)
    vals = []
    for _ in tqdm(range(n_tries), desc="δ trials"):
        idx = np.random.choice(N, sample_size, replace=False)
        sub = distance_matrix(X[idx], X[idx])
        diam = np.max(sub)
        if diam < 1e-8:
            continue
        vals.append(2.0 * delta_hyp(sub) / diam)
    return float(np.mean(vals)), float(np.std(vals))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {args.mode}")

    if args.mode == "vgg":
        print(f"\nExtracting VGG-16 features ({args.n_samples} samples) ...")
        X = extract_vgg_features(args.n_samples, args.batch_size, device)
        print(f"Feature shape : {X.shape}")
        mean_d, std_d = batched_delta_from_features(X, args.n_tries, args.sample_size)

    elif args.mode == "euclidean":
        ck = args.checkpoint or str(ROOT / DEFAULT_CKPT["euclidean"])
        print(f"\nCheckpoint : {ck}")
        X = extract_euclidean_features(ck, args.n_samples, args.batch_size, device)
        print(f"Feature shape : {X.shape}")
        mean_d, std_d = batched_delta_from_features(X, args.n_tries, args.sample_size)

    elif args.mode == "hyperbolic":
        ck = args.checkpoint or str(ROOT / DEFAULT_CKPT["hyperbolic"])
        print(f"\nCheckpoint : {ck}")
        distmat = extract_lorentz_distances(ck, args.n_samples, args.batch_size, device)
        mean_d, std_d = batched_delta(distmat, args.n_tries, args.sample_size)

    print(f"\n── Gromov δ-hyperbolicity  (CIFAR-10, mode={args.mode}) ──")
    print(f"  δ_rel = {mean_d:.4f} ± {std_d:.4f}")
    print(f"  (lower = more tree-like / hyperbolic)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",       choices=["vgg", "euclidean", "hyperbolic"], default="vgg")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--n_samples",  type=int,   default=3000,  help="Total samples to extract features from")
    parser.add_argument("--sample_size",type=int,   default=1500,  help="Sub-sample size per δ trial")
    parser.add_argument("--n_tries",    type=int,   default=10,    help="Number of repeated δ trials")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
