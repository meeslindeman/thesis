"""
IDMM two-stage training: hyperbolic or euclidean ViT.

Stage 1 (pretrain): Instance discrimination with multi-crop and CutMix.
Stage 2 (finetune): Supervised cross-entropy with the pretrained encoder.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_utils import get_dataset, get_dataloader
from utils import build_parser, cfg_from_args, set_seed, evaluate, build_scheduler

IDMM_MODELS = ["euclidean", "hyperbolic"]
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Multi-crop
# ---------------------------------------------------------------------------

class MultiCropTransform:
    """
    Returns n_global + n_local augmented views of a single PIL image.
    Global crops use a large scale range (most of the image).
    Local crops use a small scale range (zoomed-in patches), resized to `size`
    so the fixed-size ViT positional embeddings remain valid.
    """
    def __init__(self, size: int, n_global: int = 2, n_local: int = 4):
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        global_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            normalize,
        ])
        local_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            normalize,
        ])
        self.crop_transforms = [global_tf] * n_global + [local_tf] * n_local

    def __call__(self, img):
        return [t(img) for t in self.crop_transforms]


def multicrop_collate(batch):
    """
    Collate function for multi-crop batches.
    Each item is (list_of_crops, label, idx).
    Returns (list_of_crop_tensors, labels, indices) where each crop tensor is [B, C, H, W].
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
    Encoder + linear head for instance discrimination.
    Euclidean: pooled [B, D] → head.
    Hyperbolic: pooled [B, D+1] → logmap0 → drop time component → head.
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

def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5):
    """Apply CutMix to a batch. Returns (mixed_images, labels_a, labels_b, lam)."""
    B, _, H, W = images.shape
    lam = torch.distributions.Beta(alpha, alpha).sample().item()

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

def pretrain_one_epoch(model, loader, optimizer, device, use_cutmix: bool, label_smoothing: float):
    """
    Instance discrimination over all crops.
    Gradients are accumulated across crops before each optimizer step,
    keeping peak memory at one crop-batch at a time.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0

    for crops, _, instance_idx in loader:
        # crops: list of [B, C, H, W] tensors (n_global + n_local views)
        instance_idx = instance_idx.to(device, non_blocking=True).long()
        n_crops = len(crops)

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for crop in crops:
            crop = crop.to(device, non_blocking=True)
            if use_cutmix:
                crop, idx_a, idx_b, lam = cutmix_batch(crop, instance_idx)
                logits = model(crop)
                loss = (lam * loss_fn(logits, idx_a) + (1 - lam) * loss_fn(logits, idx_b)) / n_crops
            else:
                logits = model(crop)
                loss = loss_fn(logits, instance_idx) / n_crops
            loss.backward()
            batch_loss += loss.item()

        optimizer.step()
        total_loss += batch_loss * instance_idx.size(0)

    return total_loss / len(loader.dataset)


def finetune_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Model builder
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Phase 1: extract --model before building the full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="hyperbolic", choices=IDMM_MODELS)
    pre_args, _ = pre.parse_known_args()

    if pre_args.model == "euclidean":
        from vit.config import Config
    else:
        from vit_hyperbolic.config import Config

    parser = build_parser(Config)
    parser.add_argument("--model",              default=pre_args.model, choices=IDMM_MODELS)
    parser.add_argument("--stage",              choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--use_cutmix",         action="store_true", default=True)
    parser.add_argument("--n_global_crops",     type=int, default=2)
    parser.add_argument("--n_local_crops",      type=int, default=4)
    parser.add_argument("--pretrain_checkpoint", default=None)
    parser.add_argument("--checkpoint_dir",     default=None)
    parser.add_argument("--data_dir",           default="data/")
    parser.add_argument("--validation_size",    type=int, default=30)
    args = parser.parse_args()
    cfg = cfg_from_args(args, Config)

    print(f"Model: {args.model}  Stage: {args.stage}")
    print(cfg)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_start = time.perf_counter()

    vit = build_vit(args.model, cfg, device)

    if args.stage == "pretrain":
        multicrop_tf = MultiCropTransform(
            size=cfg.image_size,
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
        )

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = build_scheduler(cfg, optimizer)

        for epoch in range(cfg.epochs):
            train_loss = pretrain_one_epoch(
                model, trainloader, optimizer, device,
                use_cutmix=args.use_cutmix,
                label_smoothing=cfg.label_smoothing,
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
        model = vit

        if args.pretrain_checkpoint is not None:
            state = torch.load(args.pretrain_checkpoint, map_location=device, weights_only=True)
            encoder_state = {k: v for k, v in state["model"].items() if not k.startswith("classifier")}
            missing, unexpected = model.load_state_dict(encoder_state, strict=False)
            print(f"Loaded pretrained encoder from {args.pretrain_checkpoint}")
            print(f"  Missing (classifier, expected): {len(missing)}  Unexpected: {len(unexpected)}")

        dataset, _ = get_dataset(
            cfg.dataset, data_dir=args.data_dir,
            validation_size=args.validation_size,
        )
        dataloaders = get_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)
        trainloader = dataloaders["train"]
        testloader  = dataloaders["test"]

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        scheduler = build_scheduler(cfg, optimizer)

        for epoch in range(cfg.epochs):
            train_loss, train_acc = finetune_one_epoch(model, trainloader, optimizer, loss_fn, device)
            if scheduler is not None:
                scheduler.step()
            acc, test_loss = evaluate(model, testloader, loss_fn, device)
            print(
                f"Epoch: {epoch + 1}, "
                f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, "
                f"Test loss: {test_loss:.4f}, Test accuracy: {acc:.4f}"
            )

        if args.checkpoint_dir is not None:
            ckpt_path = Path(args.checkpoint_dir) / f"idmm_finetuned_{args.model}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"Saved finetuned model: {ckpt_path}")

    total_time = time.perf_counter() - run_start
    print(f"Total training time: {total_time:.2f}s ({total_time / 60:.2f} min)")


if __name__ == "__main__":
    main()
