#!/usr/bin/env python3
"""
Scale analysis: compare small (7-9B) vs large (70B+) vs reasoning-variant
model performance across question complexity tiers (Basic, Compound, Detailed).

Produces:
  1. Grouped bar chart (Small avg, Large avg, Reasoning avg) per tier
  2. Within-family comparison (InternVL 8B vs 78B, Qwen 7B vs 72B) per tier
  3. Printed summary table with gap analysis

Outputs saved to analysis_scripts/output/scale_analysis.pdf and .png
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "combined_results")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

SMALL_MODELS = {
    "gemma_combined.json": "Gemma-4-26B",
    "InternVideo2_5_Chat_8B_combined.json": "InternVideo2.5-8B",
    "InternVL2.5-8B_combined.json": "InternVL2.5-8B",
    "InternVL3-9B_combined.json": "InternVL3-9B",
    "internvl3_5_combined.json": "InternVL3.5-8B",
    "LLaVA-Video-7B-Qwen2_combined.json": "LLaVA-Video-7B",
    "Ovis2.5-9B_combined.json": "Ovis2.5-9B",
    "qwen2_5_7B_combined.json": "Qwen2.5-VL-7B",
    "qwen3_8B_combined.json": "Qwen3-VL-8B",
    "VideoLLaMA3-7B_combined.json": "VideoLLaMA3-7B",
}

LARGE_MODELS = {
    "InternVL2.5-78B-AWQ_combined.json": "InternVL2.5-78B",
    "qwen2_5_72B_combined.json": "Qwen2.5-VL-72B",
}

REASONING_MODELS = {
    "Ovis2.5-9B-Thinking_combined.json": "Ovis2.5-9B-Think",
    "qwen3_8B_thinking_combined.json": "Qwen3-VL-8B-Think",
    "internvl3_5_cot_combined.json": "InternVL3.5-8B-CoT",
    "internvl3_5_dot_combined.json": "InternVL3.5-8B-DoT",
}

# Within-family pairs: (small_file, large_file, family_label)
FAMILY_PAIRS = [
    ("InternVL2.5-8B_combined.json", "InternVL2.5-78B-AWQ_combined.json",
     "InternVL2.5"),
    ("qwen2_5_7B_combined.json", "qwen2_5_72B_combined.json",
     "Qwen2.5-VL"),
]

# Question type -> tier
TIER_MAP = {
    "primary_action": "Basic",
    "aggressor_identification": "Basic",
    "victim_recognition": "Basic",
    "role_identification": "Basic",
    "compound_action_aggressor": "Compound",
    "compound_action_victims": "Compound",
    "compound_aggressor_victim": "Compound",
    "compound_aggressor_action_victim": "Detailed",
    "sequence_verification": "Detailed",
}

TIER_ORDER = ["Basic", "Compound", "Detailed"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(filepath):
    """Load a combined results JSON and return the results list."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"]


def compute_tier_accuracy(results):
    """Return {tier: accuracy} for a single model's results list."""
    correct = defaultdict(int)
    total = defaultdict(int)
    for entry in results:
        tier = TIER_MAP.get(entry["question_type"])
        if tier is None:
            continue  # skip secondary question types not in our tiers
        total[tier] += 1
        if entry["is_correct"]:
            correct[tier] += 1
    return {t: (correct[t] / total[t] * 100 if total[t] > 0 else 0.0)
            for t in TIER_ORDER}


def group_accuracies(model_dict):
    """Return {tier: [acc_model1, acc_model2, ...]} for a group of models."""
    tier_accs = {t: [] for t in TIER_ORDER}
    for filename, display_name in model_dict.items():
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print("WARNING: missing file %s -- skipping %s"
                  % (filename, display_name))
            continue
        results = load_results(path)
        accs = compute_tier_accuracy(results)
        for t in TIER_ORDER:
            tier_accs[t].append(accs[t])
    return tier_accs


def safe_print(text):
    """Print ASCII-safe text."""
    if isinstance(text, str):
        print(text.encode("ascii", "replace").decode())
    else:
        print(str(text).encode("ascii", "replace").decode())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Compute per-group tier accuracies
    small_accs = group_accuracies(SMALL_MODELS)
    large_accs = group_accuracies(LARGE_MODELS)
    reasoning_accs = group_accuracies(REASONING_MODELS)

    # Means and stds per tier
    small_means = [np.mean(small_accs[t]) for t in TIER_ORDER]
    small_stds = [np.std(small_accs[t]) for t in TIER_ORDER]
    large_means = [np.mean(large_accs[t]) for t in TIER_ORDER]
    large_stds = [np.std(large_accs[t]) for t in TIER_ORDER]
    reasoning_means = [np.mean(reasoning_accs[t]) for t in TIER_ORDER]
    reasoning_stds = [np.std(reasoning_accs[t]) for t in TIER_ORDER]

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    safe_print("")
    safe_print("=" * 78)
    safe_print("Scale Analysis: Small (7-9B) vs Large (70B+) vs Reasoning variants")
    safe_print("=" * 78)
    safe_print("")
    header = "%-12s  %10s  %10s  %10s  %8s  %s" % (
        "Tier", "Small", "Large", "Reasoning", "Gap(L-S)", "Scale helps more?")
    safe_print(header)
    safe_print("-" * len(header))

    gaps = []
    for i, tier in enumerate(TIER_ORDER):
        gap = large_means[i] - small_means[i]
        gaps.append(gap)
        safe_print("%-12s  %9.1f%%  %9.1f%%  %9.1f%%  %+7.1f%%  --" % (
            tier, small_means[i], large_means[i], reasoning_means[i], gap))

    safe_print("")
    # Check if scale helps more on harder tiers
    if len(gaps) >= 3:
        if gaps[2] > gaps[0]:
            safe_print("=> Scale gap INCREASES from Basic to Detailed "
                       "(%.1f -> %.1f pp) -- scale helps more on harder tiers."
                       % (gaps[0], gaps[2]))
        else:
            safe_print("=> Scale gap DECREASES from Basic to Detailed "
                       "(%.1f -> %.1f pp) -- scale does NOT help more on "
                       "harder tiers." % (gaps[0], gaps[2]))
    safe_print("")

    # Per-model breakdown
    safe_print("--- Per-model accuracies by tier ---")
    safe_print("%-25s  %8s  %10s  %10s" % ("Model", "Basic", "Compound", "Detailed"))
    safe_print("-" * 60)
    all_models = {}
    all_models.update(SMALL_MODELS)
    all_models.update(LARGE_MODELS)
    all_models.update(REASONING_MODELS)
    for filename, display_name in sorted(all_models.items(),
                                         key=lambda x: x[1]):
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            continue
        accs = compute_tier_accuracy(load_results(path))
        safe_print("%-25s  %7.1f%%  %9.1f%%  %9.1f%%" % (
            display_name, accs["Basic"], accs["Compound"], accs["Detailed"]))
    safe_print("")

    # ------------------------------------------------------------------
    # Plot 1: Grouped bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(TIER_ORDER))
    width = 0.25

    bars_small = ax.bar(x - width, small_means, width, yerr=small_stds,
                        label="Small (7--9B)", color="#2176FF",
                        capsize=4, edgecolor="black", linewidth=0.5)
    bars_large = ax.bar(x, large_means, width, yerr=large_stds,
                        label="Large (70B+)", color="#FF8C00",
                        capsize=4, edgecolor="black", linewidth=0.5)
    bars_reason = ax.bar(x + width, reasoning_means, width,
                         yerr=reasoning_stds,
                         label="Reasoning", color="#2CA02C",
                         capsize=4, edgecolor="black", linewidth=0.5)

    # Random chance reference line (8 options -> 12.5%)
    ax.axhline(y=12.5, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=0)
    ax.text(x[-1] + width + 0.15, 12.5, "Random (12.5%)",
            va="center", ha="left", fontsize=10, color="gray")

    ax.set_xlabel("Question Tier", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title("Model Scale vs. Question Complexity", fontsize=16,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars_small, bars_large, bars_reason]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate("%.1f" % height,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Plot 2: Within-family comparison
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(1, len(FAMILY_PAIRS), figsize=(12, 5),
                              sharey=True)
    if len(FAMILY_PAIRS) == 1:
        axes = [axes]

    family_colors_small = "#A8D0E6"   # light blue
    family_colors_large = "#1B4F72"   # dark blue

    for idx, (small_file, large_file, family_name) in enumerate(FAMILY_PAIRS):
        ax2 = axes[idx]

        small_path = os.path.join(RESULTS_DIR, small_file)
        large_path = os.path.join(RESULTS_DIR, large_file)

        if not os.path.exists(small_path) or not os.path.exists(large_path):
            safe_print("WARNING: missing file for family %s" % family_name)
            continue

        small_acc = compute_tier_accuracy(load_results(small_path))
        large_acc = compute_tier_accuracy(load_results(large_path))

        small_display = SMALL_MODELS.get(small_file, small_file)
        large_display = LARGE_MODELS.get(large_file, large_file)

        s_vals = [small_acc[t] for t in TIER_ORDER]
        l_vals = [large_acc[t] for t in TIER_ORDER]

        x2 = np.arange(len(TIER_ORDER))
        w2 = 0.3

        b1 = ax2.bar(x2 - w2 / 2, s_vals, w2,
                      label=small_display, color=family_colors_small,
                      edgecolor="black", linewidth=0.5)
        b2 = ax2.bar(x2 + w2 / 2, l_vals, w2,
                      label=large_display, color=family_colors_large,
                      edgecolor="black", linewidth=0.5)

        ax2.set_title(family_name, fontsize=14, fontweight="bold")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(TIER_ORDER, fontsize=12)
        ax2.set_xlabel("Question Tier", fontsize=13)
        if idx == 0:
            ax2.set_ylabel("Accuracy (%)", fontsize=13)
        ax2.tick_params(axis="y", labelsize=11)
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(axis="y", alpha=0.3)

        # Value labels + connecting arrows + gap annotations
        for i_t in range(len(TIER_ORDER)):
            # Value labels on bars
            for bar_set, val in [(b1, s_vals[i_t]), (b2, l_vals[i_t])]:
                bar = bar_set[i_t]
                ax2.annotate("%.1f" % val,
                             xy=(bar.get_x() + bar.get_width() / 2, val),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=9)

            # Connecting arrow between paired bars
            gap_val = l_vals[i_t] - s_vals[i_t]
            small_bar = b1[i_t]
            large_bar = b2[i_t]
            sx = small_bar.get_x() + small_bar.get_width() / 2
            lx = large_bar.get_x() + large_bar.get_width() / 2
            arrow_y = max(s_vals[i_t], l_vals[i_t]) + 5
            # Draw connecting line with arrow
            ax2.annotate(
                "", xy=(lx, arrow_y), xytext=(sx, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#CC0000",
                                lw=1.5, shrinkA=2, shrinkB=2))

            # Gap annotation above the arrow
            mid_x = (sx + lx) / 2
            gap_label_y = arrow_y + 2
            ax2.annotate("%+.1f pp" % gap_val,
                         xy=(mid_x, gap_label_y), ha="center", va="bottom",
                         fontsize=11, fontweight="bold",
                         color="#CC0000" if gap_val > 0 else "#0000CC",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc="white", ec="none", alpha=0.8))

        # Print family comparison
        safe_print("--- %s family ---" % family_name)
        safe_print("%-12s  %10s  %10s  %8s" % (
            "Tier", small_display, large_display, "Gap"))
        for t in TIER_ORDER:
            g = large_acc[t] - small_acc[t]
            safe_print("%-12s  %9.1f%%  %9.1f%%  %+7.1f%%" % (
                t, small_acc[t], large_acc[t], g))
        safe_print("")

    fig2.suptitle("Within-Family Scale Comparison", fontsize=15,
                  fontweight="bold", y=1.02)
    fig2.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    # Combine both plots into a single PDF with two pages
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = os.path.join(OUTPUT_DIR, "scale_analysis.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")
    safe_print("Saved PDF: %s" % pdf_path)

    # Save PNGs for each figure separately
    png_path = os.path.join(OUTPUT_DIR, "scale_analysis.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    safe_print("Saved PNG: %s" % png_path)

    png_path2 = os.path.join(OUTPUT_DIR, "scale_analysis_families.png")
    fig2.savefig(png_path2, dpi=150, bbox_inches="tight")
    safe_print("Saved PNG: %s" % png_path2)

    pdf_path2 = os.path.join(OUTPUT_DIR, "scale_analysis_families.pdf")
    fig2.savefig(pdf_path2, bbox_inches="tight")
    safe_print("Saved PDF: %s" % pdf_path2)

    plt.close("all")
    safe_print("Done.")


if __name__ == "__main__":
    main()
