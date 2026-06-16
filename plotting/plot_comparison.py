import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODEL_ORDER = ["euclidean", "hypercore", "hexformer", "hyperbolic"]

import matplotlib.cm as cm
import numpy as np

_viridis = cm._colormaps["viridis"]
_c = [_viridis(x) for x in np.linspace(0.2, 0.8, 4)]

MODEL_COLORS = {
    "euclidean":  _c[0],
    "hypercore":  _c[1],
    "hexformer":  _c[2],
    "hyperbolic": _c[3],
}

MODEL_NAMES = {
    "euclidean":  "Euclidean",
    "hyperbolic": "Ours",
    "hypercore":  "LViT",
    "hexformer":  "HexFormer",
}

plt.rcParams.update({
    "font.size":        15,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

EPOCH_RE = re.compile(
    r"Epoch:\s*(\d+),\s*Train loss:\s*[\d.]+,\s*Train accuracy:\s*([\d.]+)"
    r",\s*Val loss:\s*[\d.]+,\s*Val accuracy:\s*([\d.]+)"
)


def parse_log(path: Path):
    epochs, train_acc, val_acc = [], [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_acc.append(float(m.group(2)))
                val_acc.append(float(m.group(3)))
    return epochs, train_acc, val_acc


def discover(root: Path):
    """Return {dataset: {model: Path}} by walking checkpoints_final."""
    found = {}
    for log in sorted(root.rglob("logs.out")):
        # expected: root/{augmentation}/{dataset}/{model}_{size}/logs.out
        parts = log.relative_to(root).parts
        if len(parts) < 4:
            continue
        dataset = parts[1]
        model   = parts[2].split("_")[0]   # e.g. "euclidean_base" → "euclidean"
        found.setdefault(dataset, {})[model] = log
    return found


def plot_dataset(dataset: str, model_logs: dict, val_only: bool, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))

    plotted = False
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        path  = model_logs.get(model)
        if path is None:
            print(f"  skipping {model} (no log found)")
            continue

        try:
            epochs, train_acc, val_acc = parse_log(path)
        except Exception as e:
            print(f"  skipping {model}: {e}")
            continue

        if not epochs:
            print(f"  skipping {model}: no epoch data in {path}")
            continue

        acc_pct  = [a * 100 for a in val_acc]
        t_pct    = [a * 100 for a in train_acc]

        name = MODEL_NAMES.get(model, model.capitalize())
        ax.plot(epochs, acc_pct, color=color, linewidth=1.8,
                linestyle="-" if val_only else "--",
                label=name if val_only else f"{name} val")
        if not val_only:
            ax.plot(epochs, t_pct, color=color, linewidth=1.8,
                    linestyle="-", alpha=0.5, label=f"{name} train")
        plotted = True

    if not plotted:
        print(f"  no data for {dataset}, skipping figure")
        plt.close(fig)
        return

    YLIMS = {"cifar10": (0, 100), "cifar100": (0, 80), "tiny_imagenet": (0, 60)}
    if dataset in YLIMS:
        ax.set_ylim(*YLIMS[dataset])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.legend(frameon=False, loc="lower right",
              fontsize=15, ncol=2 if not val_only else 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.7, zorder=0)

    fig.tight_layout()

    out = plots_dir / f"{dataset}_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  saved → {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     default="checkpoints_final",
                        help="Root directory to search for logs.out")
    parser.add_argument("--val-only", action="store_true",
                        help="Plot validation accuracy only")
    parser.add_argument("--dataset",  default=None,
                        help="Only plot a specific dataset (default: all)")
    args = parser.parse_args()

    root      = Path(args.root)
    plots_dir = Path("plotting/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_logs = discover(root)
    if not all_logs:
        print(f"No logs.out found under {root}")
        return

    for dataset, model_logs in sorted(all_logs.items()):
        if args.dataset and dataset != args.dataset:
            continue
        print(f"Plotting {dataset} ({list(model_logs.keys())})")
        plot_dataset(dataset, model_logs, args.val_only, plots_dir)


if __name__ == "__main__":
    main()
