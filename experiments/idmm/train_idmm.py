import argparse
import copy
import gc
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import asdict
from pathlib import Path
from PIL import ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_utils import get_dataset, get_dataloader
from utils import build_parser, cfg_from_args, load_yaml, set_seed, evaluate, build_scheduler

IDMM_MODELS = ["euclidean", "hyperbolic"]
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIRS = {
    "euclidean":  REPO_ROOT / "vit",
    "hyperbolic": REPO_ROOT / "vit_hyperbolic",
}
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.228, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Multi-crop
# ---------------------------------------------------------------------------

class _GaussianBlur:
    """Apply Gaussian blur with probability p (SimCLR/SwAV convention)."""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p, self.radius_min, self.radius_max = p, radius_min, radius_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(self.radius_min, self.radius_max)
        ))


def _color_distortion(s=1.0):
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return transforms.Compose([
        transforms.RandomApply([jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])


class MultiCropTransform:
    """
    Returns n_global + n_local augmented views of a single PIL image.
    All crops are resized to `size` so the ViT's positional embeddings remain valid.
    Augmentations match the paper: color distortion (s=1), grayscale, Gaussian blur.
    """
    def __init__(self, size: int, n_global: int = 2, n_local: int = 4):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        color_tf = _color_distortion(s=1.0)

        def _make_tf(scale):
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=scale),
                transforms.RandomHorizontalFlip(p=0.5),
                color_tf,
                _GaussianBlur(p=0.5),
                normalize,
            ])

        self.crop_transforms = (
            [_make_tf((0.14, 1.0))] * n_global +
            [_make_tf((0.05, 0.4))] * n_local
        )

    def __call__(self, img):
        return [t(img) for t in self.crop_transforms]


def multicrop_collate(batch):
    """
    Collate for multi-crop: each item is (list_of_crops, label, idx).
    Returns (list_of_crop_tensors [B,C,H,W], labels, indices).
    """
    crops_per_image, labels, indices = zip(*batch)
    n_crops = len(crops_per_image[0])
    crops = [
        torch.stack([crops_per_image[b][c] for b in range(len(batch))])
        for c in range(n_crops)
    ]
    return crops, torch.tensor(labels), torch.tensor(indices)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    """Returns (image, label, index) — index is the instance discrimination target."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx


# ---------------------------------------------------------------------------
# Pretrain model
# ---------------------------------------------------------------------------

class IDMMPretrainModel(nn.Module):
    """
    Encoder + linear head for instance discrimination (N_train classes).
    Hyperbolic: pooled → logmap0 → drop time component → head.
    Euclidean:  pooled directly → head.
    """
    def __init__(self, vit_model, hidden_size: int, n_instances: int):
        super().__init__()
        self.vit = vit_model
        self.head = nn.Linear(hidden_size, n_instances)
        self.is_hyperbolic = hasattr(vit_model, "manifold")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.vit.encode(x)
        if self.is_hyperbolic:
            tangent = self.vit.manifold.logmap0(pooled)
            features = tangent[..., 1:]
        else:
            features = pooled
        return self.head(features)


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5):
    """Mixup: linearly interpolate two randomly paired images."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5):
    """CutMix. lam=max(lam,1-lam) so original image always dominates (lam>=0.5)."""
    B, _, H, W = images.shape
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)

    idx = torch.randperm(B, device=images.device)
    cut_h = int(H * (1 - lam) ** 0.5)
    cut_w = int(W * (1 - lam) ** 0.5)
    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    return mixed, labels, labels[idx], lam


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

class NaNLossAbort(Exception):
    pass


def pretrain_one_epoch(model, loader, optimizer, device, use_cutmix: bool, n_global_crops: int = 2):
    """
    Instance discrimination over all crops.
    CutMix is applied to all crops (matching the paper's code).
    Loss is averaged within each group (global/local) then summed,
    so global and local crops contribute equally — matching the paper.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for crops, _, instance_idx in loader:
        instance_idx = instance_idx.to(device, non_blocking=True).long()
        n_crops = len(crops)
        n_local = n_crops - n_global_crops

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for i, crop in enumerate(crops):
            crop = crop.to(device, non_blocking=True)
            n_group = n_global_crops if i < n_global_crops else n_local
            if use_cutmix:
                crop, idx_a, idx_b, lam = cutmix_batch(crop, instance_idx)
                logits = model(crop)
                loss = (lam * loss_fn(logits, idx_a) + (1 - lam) * loss_fn(logits, idx_b)) / n_group
            else:
                logits = model(crop)
                loss = loss_fn(logits, instance_idx) / n_group
            loss.backward()
            batch_loss += loss.item()

        optimizer.step()
        total_loss += batch_loss * instance_idx.size(0)

    return total_loss / len(loader.dataset)


def finetune_one_epoch(model, loader, optimizer, loss_fn, device, epoch: int, accum_steps: int = 1):
    """Alternate Mixup (even epochs) and CutMix (odd epochs), matching the paper."""
    model.train()
    total_loss, correct = 0.0, 0
    use_mixup = (epoch % 2 == 0)

    nan_streak = 0
    nan_tolerance = 3

    optimizer.zero_grad(set_to_none=True)
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_batch(images, labels)
        else:
            images, labels_a, labels_b, lam = cutmix_batch(images, labels)

        logits = model(images)
        loss = lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)
        if torch.isfinite(loss):
            (loss / accum_steps).backward()
            nan_streak = 0
        else:
            nan_streak += 1
            print(f"  [NaN] loss is non-finite at batch {i} (streak={nan_streak})")
            if nan_streak >= nan_tolerance:
                raise NaNLossAbort(f"NaN loss for {nan_streak} consecutive batches at epoch {epoch+1}, batch {i} — aborting.")

        is_last = (i + 1 == len(loader))
        if (i + 1) % accum_steps == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if torch.isfinite(loss):
            total_loss += loss.item() * images.size(0)
            correct += (logits.detach().argmax(dim=1) == labels_a).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Model builder + positional embedding interpolation
# ---------------------------------------------------------------------------

def build_vit(model_name: str, cfg, device):
    if model_name == "euclidean":
        from vit.model import ViT
        return ViT(asdict(cfg)).to(device)
    elif model_name == "hyperbolic":
        from vit_hyperbolic.lorentz import Lorentz
        from vit_hyperbolic.model import LorentzViT
        model_cfg = asdict(cfg)
        model_cfg["manifold"] = Lorentz(cfg.curvature)
        return LorentzViT(model_cfg).to(device)
    else:
        raise ValueError(f"Unknown model '{model_name}'. IDMM supports: {IDMM_MODELS}")


def load_pretrained_encoder(model, checkpoint_path: str, cfg, device, keep_classifier: bool = False):
    """
    Load pretrained encoder weights, interpolating positional embeddings if the
    pretrain resolution differs from the current image_size (e.g. 112 → 224 or 224 → 448).

    keep_classifier: set True when continuing from a same-task fine-tuned checkpoint
                     (e.g. the 224 stage into the 448 stage) so the classifier head is reused.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if keep_classifier:
        encoder_state = dict(state["model"])
    else:
        encoder_state = {k: v for k, v in state["model"].items() if not k.startswith("classifier")}

    # Determine which key holds the position embeddings
    pos_key = None
    for candidate in ("embedding.pos_space", "embedding.position_embeddings"):
        if candidate in encoder_state:
            pos_key = candidate
            break

    # Extract and interpolate if needed
    if pos_key is not None:
        pos_weights = encoder_state.pop(pos_key)
        old_n = pos_weights.shape[1]
        new_n = (cfg.image_size // cfg.patch_size) ** 2
        if old_n != new_n:
            # Detect CLS: old_n is a perfect square → no CLS (pooling mode); otherwise has CLS
            _side = int(round(old_n ** 0.5))
            has_cls = (_side * _side != old_n)
            if has_cls:
                cls_pos = pos_weights[:, :1, :]
                patch_pos = pos_weights[:, 1:, :]
            else:
                cls_pos = None
                patch_pos = pos_weights
            old_patch_n = patch_pos.shape[1]
            old_side = int(round(old_patch_n ** 0.5))
            new_side = cfg.image_size // cfg.patch_size
            D = pos_weights.shape[2]
            pos = patch_pos.reshape(1, old_side, old_side, D).permute(0, 3, 1, 2).float()
            pos = F.interpolate(pos, size=(new_side, new_side), mode='bicubic', align_corners=False)
            patch_pos = pos.permute(0, 2, 3, 1).reshape(1, new_side * new_side, D)
            if has_cls:
                pos_weights = torch.cat([cls_pos.to(patch_pos.dtype), patch_pos], dim=1)
            else:
                pos_weights = patch_pos
            print(f"Interpolated position embeddings: {old_patch_n} → {new_side*new_side} tokens ({old_side}×{old_side} → {new_side}×{new_side})")

    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(f"Loaded pretrained encoder from {checkpoint_path}")
    print(f"  Missing (classifier, expected): {len(missing)}  Unexpected: {len(unexpected)}")

    # Set (possibly interpolated) position embeddings
    if pos_key is not None:
        if hasattr(model.embedding, 'pos_space'):
            model.embedding.pos_space.data = pos_weights.to(device)
        else:
            model.embedding.position_embeddings.data = pos_weights.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Phase 1: extract --model before building the full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="hyperbolic", choices=IDMM_MODELS)
    pre_args, _ = pre.parse_known_args()

    from config_base import Config

    yaml_overrides = load_yaml(MODEL_DIRS[pre_args.model] / "config.yml")
    parser = build_parser(Config, yaml_overrides)
    parser.add_argument("--model",               default=pre_args.model, choices=IDMM_MODELS)
    parser.add_argument("--stage",               choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--use_cutmix",          action="store_true", default=True)
    parser.add_argument("--n_global_crops",      type=int, default=2)
    parser.add_argument("--n_local_crops",       type=int, default=4)
    # Paper: pretrain at 112×112, finetune at 224×224 (positional embeddings are interpolated)
    parser.add_argument("--pretrain_image_size", type=int, default=112)
    parser.add_argument("--image_size",          type=int, default=None,
                        help="Override cfg.image_size (e.g. 448 for the high-res fine-tune stage).")
    parser.add_argument("--pretrain_checkpoint", default=None)
    parser.add_argument("--keep_classifier",     action="store_true", default=False,
                        help="Also load classifier weights from the checkpoint (use for 224→448 stage).")
    parser.add_argument("--checkpoint_dir",      default=None)
    parser.add_argument("--data_dir",            default="data/")
    parser.add_argument("--validation_size",     type=int, default=30)
    parser.add_argument("--accum_steps",         type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum_steps).")
    parser.add_argument("--save_every",          type=int, default=50,
                        help="Save a resume checkpoint every N epochs (0=disabled).")
    parser.add_argument("--resume_checkpoint",   default=None,
                        help="Path to a finetune_resume.pt to continue interrupted training.")
    args = parser.parse_args()
    cfg = cfg_from_args(args, Config)
    if args.image_size is not None:
        cfg.image_size = args.image_size

    print(f"Model: {args.model}  Stage: {args.stage}")
    print(cfg)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_start = time.perf_counter()

    if args.stage == "pretrain":
        # Build model at pretrain resolution (default 112×112, fewer tokens = faster)
        pretrain_cfg = copy.copy(cfg)
        pretrain_cfg.image_size = args.pretrain_image_size
        vit = build_vit(args.model, pretrain_cfg, device)

        multicrop_tf = MultiCropTransform(
            size=args.pretrain_image_size,
            n_global=args.n_global_crops,
            n_local=args.n_local_crops,
        )
        dataset, _ = get_dataset(
            cfg.dataset, data_dir=args.data_dir,
            validation_size=args.validation_size,
            train_transform=multicrop_tf,
        )

        n_train = len(dataset["train"])
        model = IDMMPretrainModel(vit, cfg.hidden_size, n_train).to(device)

        indexed_train = IndexedDataset(dataset["train"])
        trainloader = DataLoader(
            indexed_train, batch_size=cfg.batch_size, shuffle=True,
            drop_last=True, collate_fn=multicrop_collate,
            num_workers=8, pin_memory=True, persistent_workers=True,
        )

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = build_scheduler(cfg, optimizer)

        for epoch in range(cfg.epochs):
            train_loss = pretrain_one_epoch(
                model, trainloader, optimizer, device,
                use_cutmix=args.use_cutmix,
                n_global_crops=args.n_global_crops,
            )
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch: {epoch + 1}, Pretrain loss: {train_loss:.4f}")

        if args.checkpoint_dir is not None:
            ckpt_path = Path(args.checkpoint_dir) / "idmm_pretrained_encoder.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": vit.state_dict()}, ckpt_path)
            print(f"Saved pretrained encoder: {ckpt_path}")

    elif args.stage == "finetune":
        # Build model at full resolution (224×224); positional embeddings are interpolated on load
        vit = build_vit(args.model, cfg, device)

        optimizer = optim.AdamW(vit.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        scheduler = build_scheduler(cfg, optimizer)

        start_epoch = 0
        if args.resume_checkpoint is not None:
            state = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
            vit.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            print(f"Resumed from {args.resume_checkpoint} at epoch {start_epoch}")
        elif args.pretrain_checkpoint is not None:
            load_pretrained_encoder(vit, args.pretrain_checkpoint, cfg, device,
                                    keep_classifier=args.keep_classifier)

        dataset, _ = get_dataset(
            cfg.dataset, data_dir=args.data_dir,
            validation_size=args.validation_size,
            image_size=cfg.image_size,
        )
        dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=False)
        trainloader = dataloaders["train"]
        testloader  = dataloaders["test"]

        resume_path = Path(args.checkpoint_dir) / "finetune_resume.pt" if args.checkpoint_dir else None

        for epoch in range(start_epoch, cfg.epochs):
            try:
                train_loss, train_acc = finetune_one_epoch(vit, trainloader, optimizer, loss_fn, device, epoch, accum_steps=args.accum_steps)
            except NaNLossAbort as e:
                print(f"[FATAL] {e}")
                sys.exit(1)
            gc.collect()
            torch.cuda.empty_cache()
            if scheduler is not None:
                scheduler.step()
            acc, test_loss = evaluate(vit, testloader, loss_fn, device)
            print(
                f"Epoch: {epoch + 1}, "
                f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, "
                f"Test loss: {test_loss:.4f}, Test accuracy: {acc:.4f}"
            )
            if resume_path is not None and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                resume_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch + 1,
                    "model": vit.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                }, resume_path)
                print(f"  [ckpt] saved resume checkpoint at epoch {epoch + 1}")

        if args.checkpoint_dir is not None:
            ckpt_path = Path(args.checkpoint_dir) / f"idmm_finetuned_{args.model}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": vit.state_dict()}, ckpt_path)
            print(f"Saved finetuned model: {ckpt_path}")

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f}s ({total_time / 60:.2f} min)")


if __name__ == "__main__":
    main()
