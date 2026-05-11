import re
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def parse_slurm_log(path: str):
    epoch_re = re.compile(
        r"Epoch:\s*(\d+),\s*Train loss:\s*([\d.]+),\s*Train accuracy:\s*([\d.]+),\s*Test loss:\s*([\d.]+),\s*Test accuracy:\s*([\d.]+)"
    )
    meta_re = re.compile(r"MODEL=(\S+)\s+SIZE=(\S+)\s+EPOCHS=(\S+)\s+DATASET=(\S+)")

    epochs, train_loss, train_acc, test_loss, test_acc = [], [], [], [], []
    meta = {}

    with open(path) as f:
        for line in f:
            m = meta_re.search(line)
            if m:
                meta = {"model": m.group(1), "size": m.group(2), "dataset": m.group(4)}
            m = epoch_re.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                train_acc.append(float(m.group(3)))
                test_loss.append(float(m.group(4)))
                test_acc.append(float(m.group(5)))

    return epochs, train_loss, train_acc, test_loss, test_acc, meta


def plot(path: str, output: str | None = None):
    epochs, train_loss, train_acc, test_loss, test_acc, meta = parse_slurm_log(path)

    if not epochs:
        print("No epoch data found in file.")
        sys.exit(1)

    title = " | ".join(f"{k}={v}" for k, v in meta.items()) if meta else Path(path).name

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12)

    ax_loss.plot(epochs, train_loss, label="Train")
    ax_loss.plot(epochs, test_loss, label="Test")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, [a * 100 for a in train_acc], label="Train")
    ax_acc.plot(epochs, [a * 100 for a in test_acc], label="Test")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        out_path = Path(path).with_suffix(".png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss/acc curves from a SLURM training log.")
    parser.add_argument("log", help="Path to the .out file")
    parser.add_argument("-o", "--output", help="Output image path (default: <log>.png)")
    args = parser.parse_args()
    plot(args.log, args.output)
