import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from timm import utils
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "main"))

from data_utils import get_dataset, get_dataloader
from utils import build_parser, cfg_from_args, set_seed

MODELS = ["euclidean", "hyperbolic", "hypercore"]

from config_base import Config


def get_config_class(model: str):
    return Config


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
            hidden_channel=cfg.hidden_channel,
            out_channel=cfg.num_classes,
            mlp_hidden_size=cfg.mlp_hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout_rate,
        )


def create_metrics():
    return {"losses": utils.AverageMeter(), "top1": utils.AverageMeter(), "top5": utils.AverageMeter()}


def update_metrics(metrics, n, loss, acc1, acc5):
    metrics["losses"].update(loss.item(), n)
    metrics["top1"].update(acc1.item(), n)
    metrics["top5"].update(acc5.item(), n)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="hyperbolic", choices=MODELS)
    pre_args, _ = pre.parse_known_args()

    Config = get_config_class(pre_args.model)
    parser = build_parser(Config)
    parser.add_argument("--model", default=pre_args.model, choices=MODELS)
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--epsilon", type=float, required=True, help="FGSM perturbation magnitude")
    args = parser.parse_args()
    cfg = cfg_from_args(args, Config)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    dataset, _ = get_dataset(cfg.dataset, data_dir="data/")
    testloader = torch.utils.data.DataLoader(
        dataset["test"], batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = build_model(args.model, cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    metrics = {"clean": create_metrics(), "fgsm": create_metrics()}

    for images, labels in tqdm(testloader, desc=f"eps={args.epsilon:.5f}"):
        images, labels = images.to(device), labels.to(device).long()

        images_adv = fast_gradient_method(model_fn=model, x=images, eps=args.epsilon, norm=np.inf)

        with torch.no_grad():
            out = model(images)
            out_adv = model(images_adv)

        n = images.size(0)
        loss = loss_fn(out, labels)
        acc1, acc5 = utils.accuracy(out, labels, topk=(1, 5))
        update_metrics(metrics["clean"], n, loss, acc1, acc5)

        loss_adv = loss_fn(out_adv, labels)
        acc1_adv, acc5_adv = utils.accuracy(out_adv, labels, topk=(1, 5))
        update_metrics(metrics["fgsm"], n, loss_adv, acc1_adv, acc5_adv)

    print(f"\nModel: {args.model}  |  epsilon: {args.epsilon}")
    for attack, m in metrics.items():
        print(
            f"  {attack:5s}  Loss: {m['losses'].avg:.4f}  "
            f"Acc@1: {m['top1'].avg:.4f}  Acc@5: {m['top5'].avg:.4f}"
        )


if __name__ == "__main__":
    main()
