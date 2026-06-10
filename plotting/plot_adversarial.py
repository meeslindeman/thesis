import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

HYPERBOLIC_COLOR = "#0D9488"
EUCLIDEAN_COLOR  = "#F97316"

plt.rcParams.update({
    "font.size":        11,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

MODEL_COLORS = {
    "hyperbolic": HYPERBOLIC_COLOR,
    "euclidean":  EUCLIDEAN_COLOR,
}


def parse_log(path: str) -> dict:
    model_re  = re.compile(r"Model:\s*(\S+)\s*\|\s*epsilon:\s*([\d.]+)")
    result_re = re.compile(r"(clean|fgsm)\s+Loss:\s*[\d.]+\s+Acc@1:\s*([\d.]+)\s+Acc@5:\s*([\d.]+)")

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
                data[current_model][current_eps][m.group(1)] = {
                    "acc1": float(m.group(2)),
                    "acc5": float(m.group(3)),
                }

    return data


def plot(path: str, output: str | None = None):
    data = parse_log(path)

    fig, ax = plt.subplots(figsize=(6, 4))

    all_epsilons = None
    for model, eps_data in data.items():
        color    = MODEL_COLORS.get(model, "gray")
        epsilons = sorted(eps_data.keys())
        all_epsilons = epsilons

        fgsm_acc1  = [eps_data[e]["fgsm"]["acc1"] for e in epsilons]
        clean_acc1 = eps_data[epsilons[0]]["clean"]["acc1"]

        ax.plot(epsilons, fgsm_acc1, color=color, linewidth=1.8, marker="s", markersize=4)
        ax.axhline(clean_acc1, color=color, linewidth=1.0, linestyle="--", alpha=0.6)

    legend_handles = [
        mlines.Line2D([], [], color=c, linewidth=1.8, marker="s", markersize=4, label=m.capitalize())
        for m, c in MODEL_COLORS.items() if m in data
    ] + [
        mlines.Line2D([], [], color="gray", linewidth=1.0, linestyle="--", alpha=0.6, label="Clean"),
    ]

    ax.set_xlabel("ε")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(all_epsilons)
    ax.set_xticklabels([f"{e * 255:.1f}/255" for e in all_epsilons])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.legend(handles=legend_handles, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.7, zorder=0)

    fig.tight_layout()

    plots_dir = Path("plotting/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else plots_dir / "adversarial.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Path to adversarial log file")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    plot(args.log, args.output)
