"""Generate publication-quality figures for the paper.

Creates:
1. Ablation bar chart (component removal impact)
2. Token budget line chart (512 vs 1024 vs 1536)
3. Training loss curves (from journal data)
4. Component importance waterfall
5. Kaggle submission progression

Saves all figures as PNGs in results/figures/.
"""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_dir, "src"))
os.chdir(_project_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Use a clean style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig1_ablation_bar_chart():
    """Bar chart showing score impact of each ablation."""
    ablations = [
        ("Full System\n(baseline)", 53.79, "#2196F3"),
        ("No Fine-tuning", 46.48, "#FF9800"),
        ("No System\nPrompt", 18.80, "#F44336"),
        ("rep=1.0", 54.21, "#4CAF50"),
        ("max_tok=512", 51.47, "#FF9800"),
        ("max_tok=1536", 55.50, "#4CAF50"),
        ("No Repair", 53.63, "#9E9E9E"),
        ("Full Fine-tune\n(vs LoRA)", 53.05, "#9E9E9E"),
    ]

    names = [a[0] for a in ablations]
    scores = [a[1] for a in ablations]
    colors = [a[2] for a in ablations]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), scores, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{score:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Baseline reference line
    ax.axhline(y=53.79, color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(names) - 0.5, 54.5, "baseline = 53.79", ha="right", fontsize=9, color="#2196F3")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("Composite Score (local, 200 samples)")
    ax.set_title("Ablation Study: Component Contribution to SVG Generation Quality")
    ax.set_ylim(0, 65)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_ablation_bar_chart.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig2_token_budget():
    """Line chart showing token budget vs quality metrics."""
    tokens = [512, 1024, 1536]
    composite = [51.47, 53.79, 55.50]
    visual = [49.14, 50.92, 52.49]
    structural = [88.47, 91.85, 93.32]
    compactness = [34.48, 44.56, 49.52]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Composite score
    ax1.plot(tokens, composite, "o-", color="#2196F3", linewidth=2, markersize=8)
    for t, c in zip(tokens, composite):
        ax1.annotate(f"{c:.1f}", (t, c), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")
    ax1.set_xlabel("max_new_tokens")
    ax1.set_ylabel("Composite Score")
    ax1.set_title("Token Budget vs Composite Score")
    ax1.set_xticks(tokens)
    ax1.set_ylim(48, 58)

    # Right: Sub-metrics
    ax2.plot(tokens, visual, "s-", label="Visual Fidelity", color="#4CAF50", linewidth=2, markersize=7)
    ax2.plot(tokens, structural, "^-", label="Structural", color="#FF9800", linewidth=2, markersize=7)
    ax2.plot(tokens, compactness, "D-", label="Compactness", color="#9C27B0", linewidth=2, markersize=7)
    ax2.set_xlabel("max_new_tokens")
    ax2.set_ylabel("Score (x100)")
    ax2.set_title("Token Budget vs Sub-Metrics")
    ax2.set_xticks(tokens)
    ax2.legend()
    ax2.set_ylim(25, 100)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_token_budget.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig3_training_loss_curves():
    """Loss curves from journal data for different training approaches."""
    # LoRA r=16 (Modal, comp-only) — from Day 1-2
    lora_r16_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3200, 5000, 7000, 9189]
    lora_r16_loss = [0.71, 0.55, 0.48, 0.44, 0.41, 0.39, 0.38, 0.39, 0.36, 0.35, 0.35]

    # Full fine-tune 1.5B (Thor) — from Day 5-6
    ft_steps = [10, 1780, 2700, 3610, 4530, 5450, 6370, 7290, 8210, 9120, 10040]
    ft_loss = [1.023, 0.474, 0.432, 0.410, 0.395, 0.387, 0.458, 0.458, 0.381, 0.370, 0.397]

    # Refined (merge + full FT, LR=5e-6) — from Day 7
    ref_steps = [10, 800, 1510, 2210, 2940, 3690, 4580, 5470, 6380]
    ref_loss = [0.345, 0.325, 0.340, 0.322, 0.327, 0.337, 0.318, 0.362, 0.308]

    # Merge + r32 (kepler) — from Day 4
    merge_steps = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8500, 9532]
    merge_loss = [0.53, 0.51, 0.51, 0.48, 0.47, 0.46, 0.45, 0.40, 0.44, 0.465]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(lora_r16_steps, lora_r16_loss, "o-", label="LoRA r=16 (Modal, 1.5B)",
            color="#2196F3", linewidth=2, markersize=4)
    ax.plot(merge_steps, merge_loss, "s-", label="Merged r16 + LoRA r=32 (kepler, 1.5B)",
            color="#4CAF50", linewidth=2, markersize=4)
    ax.plot(ft_steps, ft_loss, "^-", label="Full fine-tune (Thor, 1.5B)",
            color="#FF9800", linewidth=2, markersize=4)
    ax.plot(ref_steps, ref_loss, "D-", label="Refined (merge→full FT, LR=5e-6)",
            color="#9C27B0", linewidth=2, markersize=4)

    # Loss floor annotation
    ax.axhline(y=0.34, color="red", linestyle=":", alpha=0.6, linewidth=1)
    ax.text(9500, 0.345, "LoRA loss floor ≈ 0.34", ha="right", fontsize=9, color="red")

    ax.axhline(y=0.308, color="#9C27B0", linestyle=":", alpha=0.6, linewidth=1)
    ax.text(9500, 0.298, "Refined floor ≈ 0.31", ha="right", fontsize=9, color="#9C27B0")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss Curves Across Approaches")
    ax.legend(loc="upper right")
    ax.set_ylim(0.25, 1.1)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_loss_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig4_component_importance():
    """Horizontal bar chart showing component importance (delta from baseline)."""
    components = [
        ("System Prompt", -34.99),
        ("Fine-tuning (SFT)", -7.31),
        ("Token Budget\n(512→1024)", -2.32),
        ("Training Method\n(LoRA vs Full FT)", -0.74),
        ("Repetition Penalty", +0.42),
        ("Repair Pipeline", -0.16),
        ("Token Budget\n(1024→1536)", +1.71),
    ]

    # Sort by absolute impact
    components.sort(key=lambda x: abs(x[1]), reverse=True)

    names = [c[0] for c in components]
    deltas = [c[1] for c in components]
    colors = ["#F44336" if d < -1 else "#FF9800" if d < 0 else "#4CAF50" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), deltas, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, delta in zip(bars, deltas):
        x = bar.get_width()
        ax.text(x + (0.5 if x >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"{delta:+.1f}", ha="left" if x >= 0 else "right",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Score Delta from Baseline (53.79)")
    ax.set_title("Component Importance: Impact on Composite Score")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_component_importance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig5_kaggle_progression():
    """Kaggle submission score progression over time."""
    submissions = [
        ("r16 3ep\n(Day 2)", 15.47),
        ("r32 mixed\n(Day 4)", 14.64),
        ("r32 comp\n(Day 5)", 16.87),
        ("temp=0.05\n(Day 7)", 16.42),
        ("rep=1.05\n(Day 7)", 16.51),
        ("rep=1.15\n(Day 7)", 15.88),
        ("best-of-N\n(Day 7)", 15.72),
        ("codegen\n(Day 8)", 12.26),
        ("refined\n(Day 8)", 16.26),
    ]

    names = [s[0] for s in submissions]
    scores = [s[1] for s in submissions]
    colors = ["#4CAF50" if s == max(scores) else "#F44336" if s < 14 else "#2196F3" for s in scores]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), scores, color=colors, edgecolor="white", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{score:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Best score line
    ax.axhline(y=16.87, color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(names) - 0.5, 17.1, "best = 16.87 (6th place)", ha="right", fontsize=9, color="#4CAF50")

    # Leader line
    ax.axhline(y=18.33, color="#FF9800", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(names) - 0.5, 18.55, "1st place = 18.33", ha="right", fontsize=9, color="#FF9800")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, ha="center")
    ax.set_ylabel("Kaggle Public Score")
    ax.set_title("Kaggle Submission Progression")
    ax.set_ylim(10, 20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_kaggle_progression.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def fig6_sub_metrics_radar():
    """Grouped bar chart comparing sub-metrics across key ablations."""
    ablations = {
        "Baseline": {"visual": 0.509, "structural": 0.919, "compactness": 0.446},
        "No Fine-tune": {"visual": 0.447, "structural": 0.888, "compactness": 0.223},
        "No Sys Prompt": {"visual": 0.156, "structural": 0.886, "compactness": 0.188},
        "max_tok=1536": {"visual": 0.525, "structural": 0.933, "compactness": 0.495},
    }

    metrics = ["visual", "structural", "compactness"]
    metric_labels = ["Visual Fidelity", "Structural Sim.", "Compactness"]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, vals) in enumerate(ablations.items()):
        values = [vals[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=name, color=colors[i], edgecolor="white")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Sub-Metric Comparison Across Key Ablations")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_sub_metrics.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_ablation_bar_chart()
    fig2_token_budget()
    fig3_training_loss_curves()
    fig4_component_importance()
    fig5_kaggle_progression()
    fig6_sub_metrics_radar()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
