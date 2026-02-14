"""
Generate publication-quality charts for PCA Frequency & EMD metrics.
Style: light background, strict black text, warm colour palette.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
})

WARM = ["#D94F00", "#E8751A", "#F2A541", "#F7CE68", "#FCECB4"]   # orange gradient
WARM_VAL = "#D94F00"   # dark orange  – Validation
WARM_TEST = "#F2A541"  # amber        – Test

OUT_DIR = "paper_charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
datasets = {
    "PCA Frequency Features": {
        "trials": [0, 2, 8, 5, 1],
        "val": {
            "Accuracy":  [0.7616, 0.7206, 0.6847, 0.6634, 0.6391],
            "F1":        [0.6884, 0.3218, 0.5537, 0.6869, 0.4378],
            "Precision": [0.6826, 0.8432, 0.8914, 0.5991, 0.6138],
            "Recall":    [0.6942, 0.1989, 0.4015, 0.8048, 0.3402],
            "AUROC":     [0.8070, 0.6984, 0.7667, 0.7284, 0.6583],
        },
        "test": {
            "Accuracy":  [0.7381, 0.7110, 0.7177, 0.4106, 0.6025],
            "F1":        [0.7308, 0.4804, 0.3002, 0.4567, 0.5033],
            "Precision": [0.6826, 0.8353, 0.8640, 0.6963, 0.9312],
            "Recall":    [0.7864, 0.3371, 0.1817, 0.3398, 0.3449],
            "AUROC":     [0.8576, 0.6986, 0.8075, 0.4635, 0.8201],
        },
    },
    "PCA EMD Features": {
        "trials": [0, 5, 3, 8, 4],
        "val": {
            "Accuracy":  [0.6716, 0.6553, 0.6517, 0.6462, 0.6315],
            "F1":        [0.6466, 0.6560, 0.7408, 0.6377, 0.5826],
            "Precision": [0.5461, 0.6050, 0.6499, 0.6362, 0.6678],
            "Recall":    [0.7924, 0.7165, 0.8614, 0.6392, 0.5167],
            "AUROC":     [0.7896, 0.7123, 0.6600, 0.6741, 0.6978],
        },
        "test": {
            "Accuracy":  [0.6777, 0.3880, 0.5904, 0.6023, 0.6044],
            "F1":        [0.7028, 0.3918, 0.5409, 0.4965, 0.6524],
            "Precision": [0.6026, 0.7109, 0.4318, 0.4295, 0.5757],
            "Recall":    [0.8429, 0.2704, 0.7237, 0.5882, 0.7527],
            "AUROC":     [0.7787, 0.5078, 0.6726, 0.6432, 0.6845],
        },
    },
}

metrics = ["Accuracy", "F1", "Precision", "Recall", "AUROC"]

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 – Grouped bar: Validation vs Test for each trial (per dataset)
# ══════════════════════════════════════════════════════════════════════════════
def chart_grouped_bars(ds_name, ds, tag):
    """One figure per dataset – 5 metrics, grouped bars Val vs Test."""
    trials = ds["trials"]
    n_trials = len(trials)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 4.2), sharey=True)
    fig.suptitle(f"{ds_name} — Validation vs Test (Top 5 Trials, MA=20, θ=0.49)",
                 fontsize=14, fontweight="bold", color="black", y=1.02)

    x = np.arange(n_trials)
    w = 0.35

    for i, (ax, m) in enumerate(zip(axes, metrics)):
        val_vals = ds["val"][m]
        test_vals = ds["test"][m]

        bars_v = ax.bar(x - w/2, val_vals, w, label="Validation", color=WARM_VAL, edgecolor="black", linewidth=0.6)
        bars_t = ax.bar(x + w/2, test_vals, w, label="Test",       color=WARM_TEST, edgecolor="black", linewidth=0.6)

        ax.set_title(m, fontsize=13, fontweight="bold", color="black")
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{t}" for t in trials], fontsize=10, color="black")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.tick_params(colors="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("black")

        if i == 0:
            ax.set_ylabel("Score", fontsize=12, color="black")

    axes[-1].legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"grouped_bars_{tag}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 – Radar / spider chart for best trial of each dataset
# ══════════════════════════════════════════════════════════════════════════════
def chart_radar(ds_name, ds, tag):
    """Radar chart comparing Val vs Test for the best trial (Rank 1)."""
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    val_vals = [ds["val"][m][0] for m in metrics] + [ds["val"][metrics[0]][0]]
    test_vals = [ds["test"][m][0] for m in metrics] + [ds["test"][metrics[0]][0]]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.plot(angles, val_vals, "o-", linewidth=2, color=WARM_VAL, label="Validation")
    ax.fill(angles, val_vals, alpha=0.15, color=WARM_VAL)
    ax.plot(angles, test_vals, "s-", linewidth=2, color=WARM_TEST, label="Test")
    ax.fill(angles, test_vals, alpha=0.15, color=WARM_TEST)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, color="black", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="black")
    ax.tick_params(colors="black")
    ax.spines["polar"].set_color("black")
    ax.grid(color="#cccccc", linewidth=0.5)

    trial_id = ds["trials"][0]
    ax.set_title(f"{ds_name}\nBest Trial (T{trial_id}) — Val vs Test",
                 fontsize=13, fontweight="bold", color="black", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=10, frameon=True)

    path = os.path.join(OUT_DIR, f"radar_{tag}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 – Heatmap of all metrics (Val + Test) across trials
# ══════════════════════════════════════════════════════════════════════════════
def chart_heatmap(ds_name, ds, tag):
    """Heatmap: rows = trials, columns = Val/Test × metrics."""
    trials = ds["trials"]
    col_labels = [f"Val {m}" for m in metrics] + [f"Test {m}" for m in metrics]
    data = []
    for i in range(len(trials)):
        row = [ds["val"][m][i] for m in metrics] + [ds["test"][m][i] for m in metrics]
        data.append(row)
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha="right", color="black")
    ax.set_yticks(range(len(trials)))
    ax.set_yticklabels([f"Trial {t}" for t in trials], fontsize=11, color="black")

    # Annotate cells
    for i in range(len(trials)):
        for j in range(len(col_labels)):
            val = data[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

    # Vertical separator between val and test
    ax.axvline(x=len(metrics) - 0.5, color="black", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.ax.tick_params(colors="black")
    cbar.set_label("Score", color="black", fontsize=11)

    ax.set_title(f"{ds_name} — All Metrics Heatmap (MA=20, θ=0.49)",
                 fontsize=13, fontweight="bold", color="black", pad=12)

    for spine in ax.spines.values():
        spine.set_color("black")

    path = os.path.join(OUT_DIR, f"heatmap_{tag}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 – Accuracy & F1 drop-off: val → test (lollipop / dumbbell)
# ══════════════════════════════════════════════════════════════════════════════
def chart_dropoff(ds_name, ds, tag):
    """Dumbbell chart showing how Accuracy and F1 change from Val to Test."""
    trials = ds["trials"]
    y = np.arange(len(trials))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f"{ds_name} — Validation → Test Shift",
                 fontsize=14, fontweight="bold", color="black", y=1.02)

    for ax, m in zip(axes, ["Accuracy", "F1"]):
        val_vals = ds["val"][m]
        test_vals = ds["test"][m]

        for i in range(len(trials)):
            ax.plot([val_vals[i], test_vals[i]], [i, i],
                    color="#888888", linewidth=1.5, zorder=1)
        ax.scatter(val_vals, y, color=WARM_VAL, s=90, zorder=2,
                   edgecolors="black", linewidths=0.6, label="Validation")
        ax.scatter(test_vals, y, color=WARM_TEST, s=90, zorder=2,
                   edgecolors="black", linewidths=0.6, label="Test")

        ax.set_yticks(y)
        ax.set_yticklabels([f"Trial {t}" for t in trials], fontsize=11, color="black")
        ax.set_xlabel(m, fontsize=12, color="black", fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.tick_params(colors="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#aaaaaa")

    axes[0].legend(fontsize=9, frameon=True, loc="lower left")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"dropoff_{tag}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
print("Generating paper charts …\n")

tag_map = {
    "PCA Frequency Features": "pca_freq",
    "PCA EMD Features": "pca_emd",
}

for name, ds in datasets.items():
    tag = tag_map[name]
    print(f"[{name}]")
    chart_grouped_bars(name, ds, tag)
    chart_radar(name, ds, tag)
    chart_heatmap(name, ds, tag)
    chart_dropoff(name, ds, tag)
    print()

print(f"All charts saved to ./{OUT_DIR}/")
