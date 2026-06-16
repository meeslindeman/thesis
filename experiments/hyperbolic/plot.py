import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

with open("experiments/hyperbolic/results_frechet.json") as f:
    data = json.load(f)

layers = np.arange(12)
mean_ratio   = np.array([data["per_layer"][str(i)]["mean_ratio"]   for i in layers]) * 100
max_ratio    = np.array([data["per_layer"][str(i)]["max_ratio"]    for i in layers]) * 100
mean_entropy = np.array([data["per_layer"][str(i)]["mean_entropy"] for i in layers])
mean_all     = data["aggregate"]["mean_ratio"] * 100
pearson_r    = data["aggregate"]["pearson_r_entropy_vs_ratio"]

BLUE    = "#4C72B0"
RED     = "#C44E52"
OUTLIER = "#DD8452"   # orange for layer 10
GRAY    = "0.40"

bar_colors = [OUTLIER if i == 10 else BLUE for i in layers]

fig = plt.figure(figsize=(9, 3.4))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.38)

# ── Left panel: per-layer relative error ──────────────────────────────────
ax1 = fig.add_subplot(gs[0])

ax1.bar(layers, mean_ratio, color=bar_colors, width=0.65, zorder=3, label="mean")
ax1.vlines(layers, mean_ratio, max_ratio, color=RED, lw=1.2, zorder=4, alpha=0.7)
ax1.scatter(layers, max_ratio, color=RED, s=14, zorder=5, label="worst-case token")
ax1.axhline(mean_all, ls="--", color=GRAY, lw=1, zorder=2,
            label=f"overall mean ({mean_all:.1f}%)")

# annotate layer 10
ax1.annotate("layer 10", xy=(10, mean_ratio[10]), xytext=(8.2, mean_ratio[10] + 4),
             fontsize=7, color=OUTLIER,
             arrowprops=dict(arrowstyle="-", color=OUTLIER, lw=0.8))

ax1.set_xlabel("Layer")
ax1.set_ylabel("Relative error $d(\\hat{\\mu},\\mu^*)/\\mathrm{spread}$ (%)")
ax1.set_xticks(layers)
ax1.set_xlim(-0.6, 11.6)
ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)

# ── Right panel: entropy vs mean relative error (per layer) ───────────────
ax2 = fig.add_subplot(gs[1])

scatter_colors = [OUTLIER if i == 10 else BLUE for i in layers]
ax2.scatter(mean_entropy, mean_ratio, c=scatter_colors, s=28, zorder=4)

# label layer 10 and a couple of neighbours
for i, label in [(10, "10"), (0, "0"), (11, "11")]:
    ax2.annotate(str(i), xy=(mean_entropy[i], mean_ratio[i]),
                 xytext=(3, 3), textcoords="offset points",
                 fontsize=6.5, color=scatter_colors[i])

# flat regression line to show weak correlation
m, b = np.polyfit(mean_entropy, mean_ratio, 1)
x_line = np.linspace(mean_entropy.min() - 0.1, mean_entropy.max() + 0.1, 50)
ax2.plot(x_line, m * x_line + b, color=GRAY, lw=1, ls="--", zorder=3)
ax2.text(0.97, 0.06, f"per-token $r = {pearson_r:.2f}$", transform=ax2.transAxes,
         ha="right", va="bottom", fontsize=7.5, color=GRAY)

ax2.set_xlabel("Mean attention entropy")
ax2.set_ylabel("Mean relative error (%)")
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("experiments/hyperbolic/out/value_aggregation.pdf",
            dpi=200, bbox_inches="tight")
fig.savefig("experiments/hyperbolic/out/value_aggregation.png",
            dpi=200, bbox_inches="tight")
print("Saved value_aggregation.{pdf,png}")
