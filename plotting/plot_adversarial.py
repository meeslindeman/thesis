import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np

plt.rcParams.update({
    "font.size":        15,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

MODEL_ORDER = ["euclidean", "hypercore", "hexformer", "hyperbolic"]

MODEL_NAMES = {
    "euclidean":  "Euclidean",
    "hyperbolic": "Ours",
    "hypercore":  "LViT",
    "hexformer":  "HexFormer",
}

_cividis = plt.colormaps["cividis"]
_c = [_cividis(x) for x in np.linspace(0.1, 0.9, 4)]
MODEL_COLORS = {
    "euclidean":  _c[0],
    "hypercore":  _c[1],
    "hexformer":  _c[2],
    "hyperbolic": _c[3],
}


def parse_log(path: str) -> dict:
    model_re  = re.compile(r"Model:\s*(\S+)\s*\|\s*epsilon:\s*([\d.]+)")
    result_re = re.compile(r"(clean|fgsm)\s+Loss:\s*[\d.]+\s+Acc@1:\s*([\d.]+)")

    data = {}
    current_model, current_eps = None, None

    with open(path) as f:
        for line in f:
            m = model_re.search(line)
            if m:
                current_model = m.group(1)
                current_eps   = float(m.group(2))
                data.setdefault(current_model, {})[current_eps] = {}
                continue
            m = result_re.search(line)
            if m and current_model is not None:
                data[current_model][current_eps][m.group(1)] = float(m.group(2))

    return data


def plot(path: str, output: str | None = None):
    data     = parse_log(path)
    present  = [m for m in MODEL_ORDER if m in data]
    n_models = len(present)

    # All epsilon values (assumed consistent across models)
    epsilons = sorted(next(iter(data.values())).keys())
    # Groups: clean first, then one per epsilon
    groups      = ["Clean"] + [f"{e * 255:.1f}/255" for e in epsilons]
    n_groups    = len(groups)

    bar_w = 0.7 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, model in enumerate(present):
        color   = MODEL_COLORS[model]
        eps_data = data[model]
        clean_acc = eps_data[epsilons[0]]["clean"]
        fgsm_accs = [eps_data[e]["fgsm"] for e in epsilons]
        values    = [clean_acc] + fgsm_accs

        ax.bar(x + offsets[i], values, width=bar_w * 0.9,
               color=color, label=MODEL_NAMES.get(model, model.capitalize()),
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 80)
    ax.set_xlabel("ε")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=n_models, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.7, zorder=0)

    fig.tight_layout()

    plots_dir = Path("plotting/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else plots_dir / "adversarial.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Path to adversarial log file")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    plot(args.log, args.output)
