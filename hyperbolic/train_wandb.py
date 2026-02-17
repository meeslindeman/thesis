import torch
import torch.nn as nn
import torch.optim as optim

from hyperbolic.model import LorentzViT
from hyperbolic.lorentz import Lorentz


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, *, amp_dtype="none", wandb_run=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.wandb_run = wandb_run

        if amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16
        elif amp_dtype == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

    def train(self, trainloader, testloader, epochs):
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            acc, test_loss = self.evaluate(testloader)

            print(
                f"Epoch: {i+1}, "
                f"Train loss: {train_loss:.4f}, "
                f"Test loss: {test_loss:.4f}, "
                f"Accuracy: {acc:.4f}"
            )

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "epoch": i + 1,
                        "train/loss": train_loss,
                        "test/loss": test_loss,
                        "test/acc": acc,
                    }
                )

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0.0

        for images, labels in trainloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.amp_dtype is None:
                logits, _ = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss.backward()
            else:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    logits, _ = self.model(images)
                    loss = self.loss_fn(logits, labels)
                loss.backward()

            self.optimizer.step()
            total_loss += loss.item() * images.size(0)

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0

        for images, labels in testloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.amp_dtype is None:
                logits, _ = self.model(images)
            else:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    logits, _ = self.model(images)

            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        return correct / len(testloader.dataset), total_loss / len(testloader.dataset)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", type=str, choices=["bf16", "fp16", "none"], default="none")

    # wandb (enable by default; set --wandb 0 to disable)
    parser.add_argument("--wandb", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="hyperbolic-vit")
    parser.add_argument("--wandb_entity", type=str, default="hyperbolic-vit-team")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")

    # model (LorentzViT)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=48)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=192)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--qkv_bias", type=int, default=1)

    # hyperbolic specifics
    parser.add_argument("--curvature", type=float, default=0.1)
    parser.add_argument("--head_mode", type=str, default="full_midpoint")
    parser.add_argument("--concat_mode", type=str, default="direct")
    parser.add_argument("--compute_scores", type=str, default="lorentz_inner")
    parser.add_argument("--reset_params", type=str, default="lorentz_kaiming")
    parser.add_argument("--skip_connection", action="store_true", default=False)
    parser.add_argument("--a_default", type=float, default=0.0)

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.qkv_bias = bool(args.qkv_bias)
    args.wandb = bool(args.wandb)

    return args


def main():
    args = parse_args()
    print(args)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )

    from cifar10_utils import get_cifar10, get_dataloader

    dataset = get_cifar10(data_dir="data/", validation_size=5000)
    dataloaders = get_dataloader(dataset, args.batch_size)
    trainloader = dataloaders["train"]
    testloader = dataloaders["test"]

    model_cfg = {
        "patch_size": args.patch_size,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "mlp_dim": args.mlp_dim,
        "dropout_rate": args.dropout_rate,
        "initializer_range": args.initializer_range,
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "num_channels": args.num_channels,
        "qkv_bias": args.qkv_bias,
        "curvature": args.curvature,
        "manifold": Lorentz(args.curvature),
        "head_mode": args.head_mode,
        "concat_mode": args.concat_mode,
        "compute_scores": args.compute_scores,
        "reset_params": args.reset_params,
        "a_default": args.a_default,
        "skip_connection": args.skip_connection,
    }

    model = LorentzViT(model_cfg)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=args.device,
        amp_dtype=args.amp,
        wandb_run=wandb_run,
    )
    trainer.train(trainloader, testloader, args.epochs)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
