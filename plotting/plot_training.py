import re
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

TRAIN_COLOR = "#2563EB"
VAL_COLOR   = "#DC2626"

plt.rcParams.update({
    "font.size":        11,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})


def parse_slurm_log(path: str):
    epoch_re = re.compile(
        r"Epoch:\s*(\d+),\s*Train loss:\s*([\d.]+),\s*Train accuracy:\s*([\d.]+),\s*Val loss:\s*([\d.]+),\s*Val accuracy:\s*([\d.]+)"
    )
    meta_re = re.compile(r"MODEL=(\S+)\s+SIZE=(\S+)\s+DATASET=(\S+)\s+AUGMENTATION=(\S+)")

    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    meta = {}

    with open(path) as f:
        for line in f:
            m = meta_re.search(line)
            if m:
                meta = {"model": m.group(1), "size": m.group(2), "dataset": m.group(3), "augmentation": m.group(4)}
            m = epoch_re.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_acc.append(float(m.group(3)))
                val_loss.append(float(m.group(4)))
                val_acc.append(float(m.group(5)))

    return epochs, train_loss, train_acc, val_loss, val_acc, meta


def plot(path: str, output: str | None = None):
    epochs, train_loss, train_acc, val_loss, val_acc, meta = parse_slurm_log(path)

    if not epochs:
        print("No epoch data found in file.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(epochs, [a * 100 for a in train_acc], color=TRAIN_COLOR, linewidth=1.8, label="Train")
    ax.plot(epochs, [a * 100 for a in val_acc],  color=VAL_COLOR,   linewidth=1.8, label="Val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    if meta.get("dataset") == "cifar10":
        ax.set_ylim(0, 100)
    if meta.get("dataset") == "cifar100":
        ax.set_ylim(0, 80)
    if meta.get("dataset") == "tiny_imagenet":
        ax.set_ylim(0, 80)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.7, zorder=0)

    fig.tight_layout()

    plots_dir = Path("plotting/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if output:
        out_path = Path(output)
    elif meta:
        stem = f"{meta['model']}_{meta['size']}_{meta['dataset']}_{meta['augmentation']}"
        out_path = plots_dir / f"{stem}.png"
    else:
        out_path = plots_dir / (Path(path).stem + ".png")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot accuracy curves from a SLURM training log.")
    parser.add_argument("log", help="Path to the .out file")
    parser.add_argument("-o", "--output", help="Output image path (default: plotting/train_plots/<meta>.png)")
    args = parser.parse_args()
    plot(args.log, args.output)
