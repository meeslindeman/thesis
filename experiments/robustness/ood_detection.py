import argparse
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "main"))

from data_utils import get_dataset, DATASET_REGISTRY
from utils import set_seed

MODELS = ["euclidean", "hyperbolic", "hypercore"]

from config_base import Config


# ── OOD metric helpers ─────────────────────────────────────────────────────────

def _stable_cumsum(arr):
    out = np.cumsum(arr, dtype=np.float64)
    if not np.allclose(out[-1], np.sum(arr, dtype=np.float64)):
        raise RuntimeError("cumsum unstable")
    return out


def _fpr_at_recall(y_true, y_score, recall_level=0.95):
    y_true = y_true == 1
    idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_score, y_true = y_score[idx], y_true[idx]
    threshold_idxs = np.r_[np.where(np.diff(y_score))[0], y_true.size - 1]
    tps = _stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    recall = tps / tps[-1]
    last = tps.searchsorted(tps[-1])
    sl = slice(last, None, -1)
    recall, fps = np.r_[recall[sl], 1], np.r_[fps[sl], 0]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / np.sum(~y_true)


def get_measures(pos, neg, recall_level=0.95):
    examples = np.concatenate([np.array(pos), np.array(neg)])
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] = 1
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = _fpr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr


def print_measures(auroc, aupr, fpr, name="", recall_level=0.95):
    print(f"  [{name}]  FPR{int(100 * recall_level)} / AUROC / AUPR")
    print("  & {:.2f} & {:.2f} & {:.2f}".format(100 * fpr, 100 * auroc, 100 * aupr))


# ── Model building ─────────────────────────────────────────────────────────────

def build_model(model: str, cfg):
    if model == "euclidean":
        from vit.model import ViT
        return ViT(asdict(cfg))

    elif model == "hyperbolic":
        from vit_hyperbolic.lorentz import Lorentz
        from vit_hyperbolic.model import LorentzViT
        model_cfg = asdict(cfg)
        model_cfg["manifold"] = Lorentz(cfg.curvature)
        return LorentzViT(model_cfg)

    elif model == "hypercore":
        from vit_hypercore.model import LViT
        from vit_hypercore.manifold.lorentz import Lorentz
        manifold = Lorentz(cfg.curvature)
        return LViT(
            manifold_in=manifold,
            manifold_hidden=manifold,
            manifold_out=manifold,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_hidden_layers,
            in_channel=cfg.num_channels,
            hidden_channel=cfg.hidden_size // cfg.num_attention_heads,
            out_channel=cfg.num_classes,
            mlp_hidden_size=cfg.mlp_dim + 1,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout_rate,
        )


# ── Score extraction ───────────────────────────────────────────────────────────

@torch.no_grad()
def get_in_dist_scores(model, loader, device):
    scores, right_scores, wrong_scores = [], [], []
    for data, targets in tqdm(loader, desc="In-distribution"):
        data = data.to(device)
        smax = F.softmax(model(data), dim=1).cpu().numpy()
        scores.append(-np.max(smax, axis=1))
        preds = np.argmax(smax, axis=1)
        tgt = targets.numpy().squeeze()
        right_mask = preds == tgt
        if right_mask.any():
            right_scores.append(-np.max(smax[right_mask], axis=1))
        if (~right_mask).any():
            wrong_scores.append(-np.max(smax[~right_mask], axis=1))
    return (
        np.concatenate(scores),
        np.concatenate(right_scores) if right_scores else np.array([]),
        np.concatenate(wrong_scores) if wrong_scores else np.array([]),
    )


@torch.no_grad()
def get_ood_scores(model, loader, device, max_batches=None):
    scores = []
    for i, (data, _) in enumerate(tqdm(loader, desc="OOD", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        data = data.to(device)
        smax = F.softmax(model(data), dim=1).cpu().numpy()
        scores.append(-np.max(smax, axis=1))
    return np.concatenate(scores)


# ── OOD dataset helpers ────────────────────────────────────────────────────────

def make_ood_transform(cfg):
    meta = DATASET_REGISTRY[cfg.dataset]
    size = meta["image_size"]
    return trn.Compose([
        trn.Resize(size),
        trn.CenterCrop(size),
        trn.ToTensor(),
        trn.Normalize(meta["mean"], meta["std"]),
    ])


def make_ood_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def _cfg_from_checkpoint(path: str) -> tuple:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in state:
        raise RuntimeError(f"Checkpoint {path} has no saved config.")
    init_names = {f.name for f in fields(Config) if f.init}
    cfg = Config(**{k: v for k, v in state["config"].items() if k in init_names})
    return cfg, state


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", required=True, choices=MODELS)
    pre.add_argument("--checkpoint", required=True)
    pre_args, _ = pre.parse_known_args()

    cfg, state = _cfg_from_checkpoint(pre_args.checkpoint)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--textures_dir", default=None, help="DTD Textures root (ImageFolder layout)")
    parser.add_argument("--svhn_dir", default=None, help="SVHN root; downloaded automatically on first run")
    parser.add_argument("--places365_dir", default=None, help="Places365 root (ImageFolder layout)")
    args = parser.parse_args()

    cfg.batch_size = args.batch_size
    cfg.device = args.device
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    dataset, _ = get_dataset(cfg.dataset, data_dir="data/")
    test_loader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = build_model(args.model, cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    ood_transform = make_ood_transform(cfg)
    ood_num_examples = len(dataset["test"]) // 5

    print(f"\nModel: {args.model}  |  Dataset: {cfg.dataset}  |  Score: MSP")

    in_score, right_score, wrong_score = get_in_dist_scores(model, test_loader, device)
    num_right, num_wrong = len(right_score), len(wrong_score)
    print(f"Error Rate: {100 * num_wrong / (num_wrong + num_right):.2f}%")

    if len(wrong_score) > 0 and len(right_score) > 0:
        print("\n── Error Detection ──")
        auroc, aupr, fpr = get_measures(wrong_score, right_score)
        print_measures(auroc, aupr, fpr, name=args.model)

    auroc_list, aupr_list, fpr_list = [], [], []
    max_batches = ood_num_examples // cfg.batch_size

    def evaluate_ood(name, ood_data):
        out_score = get_ood_scores(model, make_ood_loader(ood_data, cfg.batch_size), device, max_batches)
        auroc, aupr, fpr = get_measures(-in_score, -out_score)
        auroc_list.append(auroc)
        aupr_list.append(aupr)
        fpr_list.append(fpr)
        print(f"\n── {name} ──")
        print_measures(auroc, aupr, fpr, name=args.model)

    if args.textures_dir:
        evaluate_ood("Textures", dset.ImageFolder(root=args.textures_dir, transform=ood_transform))

    if args.svhn_dir:
        evaluate_ood("SVHN", dset.SVHN(root=args.svhn_dir, split="test", transform=ood_transform, download=True))

    if args.places365_dir:
        evaluate_ood("Places365", dset.Places365(
            root=args.places365_dir, split="val", small=True,
            transform=ood_transform, download=False,
        ))

    if auroc_list:
        print("\n── Mean OOD Results ──")
        print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), name=args.model)


if __name__ == "__main__":
    main()
