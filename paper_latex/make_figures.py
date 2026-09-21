"""Generate all figures for the LaTeX manuscript from verified project results.

Every number here is taken directly from paper_latex/main.tex / docs/01-ket-qua-ablation.md
(already hand-verified multiple times earlier in this project) or read directly from the
underlying results CSVs. Do not adjust numbers here without re-checking the source of truth.

Run:
    python paper_latex/make_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLOR_CORRECT = "#2E7D32"
COLOR_HALLUC = "#C62828"
COLOR_CONFUSABLE = "#EF6C00"


# ---------------------------------------------------------------------------
# Figure 1: Phase 0 — zero-shot vs. CE-only fine-tune, both datasets
# Source: results/kaggle_run_2_lora_ceonly/summary_before_after.csv (Run 1),
# main.tex Table 4 (Section 4.1)
# ---------------------------------------------------------------------------
def fig1_phase0():
    datasets = ["Kaggle-BD (78 classes)", "RxHandBD (~1,430 labels)"]
    conditions = ["Zero-shot", "CE-only fine-tune"]
    # rates as fractions: [correct, hallucination_far_off, confusable_wrong_drug]
    data = {
        ("Kaggle-BD (78 classes)", "Zero-shot"): [22.95, 60.77, 0.38],
        ("Kaggle-BD (78 classes)", "CE-only fine-tune"): [87.95, 5.77, 0.64],
        ("RxHandBD (~1,430 labels)", "Zero-shot"): [19.55, 60.90, 1.79],
        ("RxHandBD (~1,430 labels)", "CE-only fine-tune"): [46.64, 22.15, 7.26],
    }
    categories = ["Correct", "Hallucination\n(far-off)", "Confusable\n(wrong drug)"]
    colors = [COLOR_CORRECT, COLOR_HALLUC, COLOR_CONFUSABLE]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    x = np.arange(len(categories))
    width = 0.35

    for ax, dataset in zip(axes, datasets):
        for i, cond in enumerate(conditions):
            vals = data[(dataset, cond)]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=cond,
                           color=colors, alpha=(1.0 if i == 1 else 0.45),
                           edgecolor="black", linewidth=0.6)
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(dataset, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Rate (%)")

    # custom legend: condition (alpha) x category (color)
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="gray", alpha=0.45, edgecolor="black", label="Zero-shot"),
        Patch(facecolor="gray", alpha=1.0, edgecolor="black", label="CE-only fine-tune"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "fig1_phase0_zeroshot_vs_finetune.pdf")
    plt.savefig(out)
    plt.close()
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 2: Phase 1 exploratory sweep — confusable_wrong_drug rate across
# all 7 configurations (kernel 03 + 04), single unseeded run each.
# Source: main.tex Tables 4-6 (Section 3.5.1 / 4.2)
# ---------------------------------------------------------------------------
def fig2_phase1_sweep():
    configs = [
        "CE-only\nbaseline",
        "Severity-only\n(kernel 03)",
        "Margin-only\n$\\lambda$=1.0",
        "Severity+margin\n$\\lambda$=1.0",
        "Margin\n$\\lambda$=2.0",
        "Margin\n$\\lambda$=3.0",
        "Margin $\\lambda$=2.0,\n5 epochs",
        "Margin $\\lambda$=2.0 +\nseverity $\\beta$=0.3\n(selected)",
    ]
    rates = [7.44, 7.89, 6.64, 7.44, 6.01, 5.29, 6.55, 6.01]
    p_holm = [None, 0.946, 0.573, 1.000, 0.054, 0.0008, 0.573, 0.057]
    survives = [False, False, False, False, False, True, False, False]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(configs))
    bar_colors = ["#616161"] + [
        ("#2E7D32" if s else "#EF6C00") for s in survives[1:]
    ]
    bars = ax.bar(x, rates, color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.axhline(7.44, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("confusable_wrong_drug rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=7)
    ax.set_ylim(0, 9)

    for xi, r, p in zip(x, rates, p_holm):
        label = f"{r:.2f}%"
        if p is not None:
            label += f"\n$p_{{holm}}$={p:.3f}" if p >= 0.001 else f"\n$p_{{holm}}$={p:.4f}"
        ax.text(xi, r + 0.15, label, ha="center", va="bottom", fontsize=6.3)

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="#616161", edgecolor="black", label="CE-only baseline"),
        Patch(facecolor="#EF6C00", edgecolor="black", label="Does not survive Holm–Bonferroni"),
        Patch(facecolor="#2E7D32", edgecolor="black", label="Survives Holm–Bonferroni ($\\alpha$=0.05)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7, frameon=False)
    ax.set_title(
        "Phase 1 (exploratory, single unseeded run per configuration) — RxHandBD, n=1,115",
        fontsize=9,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_phase1_exploratory_sweep.pdf")
    plt.savefig(out)
    plt.close()
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 3: Forest plot of Phase 2 pooled effect sizes (risk differences with
# 95% CI). Source: main.tex Table 11 / Section 5.1.
# ---------------------------------------------------------------------------
def fig3_forest_plot():
    metrics = [
        "confusable_wrong_drug",
        "correct",
        "hallucination_far_off",
    ][::-1]
    risk_diff = [-0.90, -2.01, +2.15][::-1]
    ci_lo = [-1.34, -2.80, +1.39][::-1]
    ci_hi = [-0.46, -1.22, +2.92][::-1]
    p_vals = ["p=0.000077", "p=0.000001", "p<0.000001"][::-1]

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    y = np.arange(len(metrics))
    colors = ["#EF6C00" if rd < 0 else "#C62828" for rd in risk_diff]

    for yi, rd, lo, hi, c in zip(y, risk_diff, ci_lo, ci_hi, colors):
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(rd, yi, "o", color=c, markersize=8, markeredgecolor="black", markeredgewidth=0.6)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Risk difference, winning config − CE-only (percentage points)")
    ax.set_xlim(-3.5, 3.5)

    for yi, rd, hi, p in zip(y, risk_diff, ci_hi, p_vals):
        ax.text(3.6, yi, f"{rd:+.2f} pp, {p}", va="center", fontsize=8)

    ax.set_title(
        "Phase 2 confirmatory pooled effect sizes (5 seeds, n=5,575), 95% CI",
        fontsize=9,
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_phase2_forest_plot.pdf")
    plt.savefig(out)
    plt.close()
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 4: Destination of resolved confusable_wrong_drug errors (pooled, 5
# seeds). Source: main.tex Table 12 / Section 4.5.
# ---------------------------------------------------------------------------
def fig4_destination_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw={"width_ratios": [1, 1.3]})

    # left: overall confusable pool -> resolved vs still-confusable
    total_confusable = 399
    resolved = 103
    still = total_confusable - resolved
    ax = axes[0]
    ax.pie(
        [resolved, still],
        labels=[f"Resolved\n{resolved}/{total_confusable} (25.8%)", f"Still confusable\n{still}/{total_confusable} (74.2%)"],
        colors=["#2E7D32", "#9E9E9E"],
        autopct=None,
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 0.6},
        textprops={"fontsize": 7.5},
    )
    ax.set_title("All confusable_wrong_drug errors\n(CE-only, pooled 5 seeds, n=399)", fontsize=8.5)

    # right: of the resolved 103, where did they go
    ax2 = axes[1]
    dest_labels = ["Fully corrected\n(36, 35.0%)", "Minor OCR noise\n(37, 35.9%)", "Hallucination\n(30, 29.1%)"]
    dest_vals = [36, 37, 30]
    dest_colors = ["#2E7D32", "#EF6C00", "#C62828"]
    ax2.pie(
        dest_vals,
        labels=dest_labels,
        colors=dest_colors,
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 0.6},
        textprops={"fontsize": 7.5},
    )
    ax2.set_title("Destination of the 103 resolved errors\nunder the winning configuration", fontsize=8.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_destination_breakdown.pdf")
    plt.savefig(out)
    plt.close()
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure 5: Two-phase (exploratory -> confirmatory) experimental design
# schematic. Purely a diagram, no data points (illustrates methodology).
# ---------------------------------------------------------------------------
def fig5_design_schematic():
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.3, wrap=True)

    # Phase 1
    box(0.2, 2.1, 3.0, 1.5, "Phase 1: Exploratory\n7 configs × 1 unseeded run\n(kernels 03, 04)\nRxHandBD, n=1,115", "#FFE0B2")
    box(3.6, 2.1, 2.6, 1.5, "Holm–Bonferroni\nsensitivity check\n(7-way family)", "#FFE0B2")
    box(6.6, 2.1, 3.2, 1.5, "Select ONE config:\nmargin $\\lambda$=2.0 +\nseverity $\\beta$=0.3", "#FFCC80")

    # Phase 2
    box(0.2, 0.2, 3.6, 1.5, "Phase 2: Confirmatory\n5 fully-seeded, paired\nreplications (kernel 06)", "#C8E6C9")
    box(4.2, 0.2, 2.8, 1.5, "Per-seed + pooled\nexact McNemar tests", "#C8E6C9")
    box(7.4, 0.2, 2.4, 1.5, "Effect sizes\nwith 95% CI\n(primary evidence)", "#A5D6A7")

    # arrows
    for (x0, x1, y) in [(3.2, 3.6, 2.85), (6.2, 6.6, 2.85), (3.8, 4.2, 0.95), (7.0, 7.4, 0.95)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("", xy=(8.2, 1.7), xytext=(8.2, 2.1),
                arrowprops=dict(arrowstyle="->", lw=1.0))

    ax.text(0.2, 3.85, "Purpose: hypothesis generation only, NOT confirmatory evidence", fontsize=6.8, style="italic")
    ax.text(0.2, 1.85, "Purpose: primary evidence for this paper's central claim", fontsize=6.8, style="italic")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_two_phase_design.pdf")
    plt.savefig(out)
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    fig1_phase0()
    fig2_phase1_sweep()
    fig3_forest_plot()
    fig4_destination_breakdown()
    fig5_design_schematic()
    print("All figures written to", OUT_DIR)
