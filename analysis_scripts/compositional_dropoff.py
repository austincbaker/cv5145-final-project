#!/usr/bin/env python3
"""Plot how each model's accuracy degrades from Basic -> Compound -> Detailed tiers."""

import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "combined_results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

ALL_MODEL_FILES = {
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
    "qwen2_5_72B_combined.json": "Qwen2.5-VL-72B",
    "gpt_5-1_combined.json": "GPT5.1",
    "qwen3_8B_thinking_combined.json": "Qwen3-VL-8B-Think",
    "internvl3_5_cot_combined.json": "InternVL3.5-8B-CoT",
    "internvl3_5_dot_combined.json": "InternVL3.5-8B-DoT",
}

# 5 highlighted models -- only these get plotted as lines
HIGHLIGHT_MODELS = [
    ("qwen2_5_72B_combined.json", "Qwen2.5-VL-72B"),
    ("VideoLLaMA3-7B_combined.json", "VideoLLaMA3-7B"),
    ("internvl3_5_dot_combined.json", "InternVL3.5-8B-DoT"),
    ("qwen3_8B_combined.json", "Qwen3-VL-8B"),
    ("Ovis2.5-9B_combined.json", "Ovis2.5-9B"),
]

HIGHLIGHT_COLORS = [
    "#1f77b4",  # Qwen2.5-VL-72B   -- blue
    "#d62728",  # VideoLLaMA3-7B    -- red
    "#2ca02c",  # InternVL3.5-DoT   -- green
    "#ff7f0e",  # Qwen3-VL-8B       -- orange
    "#9467bd",  # Ovis2.5-9B        -- purple
]

HIGHLIGHT_MARKERS = ["o", "s", "^", "D", "v"]

REASONING_MODELS = {
    "Ovis2.5-9B-Think",
    "Qwen3-VL-8B-Think",
    "InternVL3.5-8B-CoT",
    "InternVL3.5-8B-DoT",
}

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

TIERS = ["Basic", "Compound", "Detailed"]


def load_model_tier_accuracy(filepath):
    """Return {tier: accuracy%} for a single model file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for entry in data["results"]:
        qtype = entry.get("question_type", "")
        tier = TIER_MAP.get(qtype)
        if tier is None:
            continue
        counts[tier]["total"] += 1
        if entry.get("is_correct", False):
            counts[tier]["correct"] += 1

    acc = {}
    for tier in TIERS:
        total = counts[tier]["total"]
        if total > 0:
            acc[tier] = 100.0 * counts[tier]["correct"] / total
        else:
            acc[tier] = None
    return acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load ALL 16 models (needed for gray band)
    all_accs = {}  # display_name -> {tier: accuracy}
    for fname, display_name in ALL_MODEL_FILES.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(fpath):
            print("WARN: missing file %s -- skipping %s" % (fname, display_name))
            continue
        all_accs[display_name] = load_model_tier_accuracy(fpath)

    if not all_accs:
        print("FAIL: no model files found")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Print drop rates (Basic - Detailed), sorted steepest first
    # -------------------------------------------------------------------
    drops = []
    for name, accs in all_accs.items():
        basic = accs.get("Basic")
        detailed = accs.get("Detailed")
        if basic is not None and detailed is not None:
            drops.append((name, basic, detailed, basic - detailed))

    drops.sort(key=lambda x: -x[3])  # steepest drop first

    print("")
    print("Compositional drop-off: Basic -> Detailed")
    print("-" * 62)
    print("%-25s  %8s  %8s  %8s" % ("Model", "Basic", "Detailed", "Drop"))
    print("-" * 62)
    for name, basic, detailed, drop in drops:
        print("%-25s  %7.2f%%  %7.2f%%  %7.2f%%" % (name, basic, detailed, drop))
    print("-" * 62)
    print("")

    # -------------------------------------------------------------------
    # Compute gray band: min/max across all 16 models at each tier
    # -------------------------------------------------------------------
    tier_mins = []
    tier_maxs = []
    for tier in TIERS:
        vals = [
            accs[tier] for accs in all_accs.values()
            if accs.get(tier) is not None
        ]
        tier_mins.append(min(vals))
        tier_maxs.append(max(vals))

    # -------------------------------------------------------------------
    # Plot -- 5 highlighted models only, with gray band for context
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    x_pos = np.arange(len(TIERS))

    # Gray band showing range of all 16 models
    ax.fill_between(
        x_pos, tier_mins, tier_maxs,
        color="#cccccc", alpha=0.4,
        label="All 16 models (min--max)",
    )

    # Random chance baseline (8 options)
    ax.axhline(
        y=12.5, color="gray", linestyle=":", linewidth=1.5,
        label="Random chance (12.5%)",
    )

    # Plot the 5 highlighted models
    for idx, (fname, display_name) in enumerate(HIGHLIGHT_MODELS):
        if display_name not in all_accs:
            print("WARN: %s not loaded -- skipping" % display_name)
            continue

        accs = all_accs[display_name]
        ys = [accs.get(t) for t in TIERS]
        if any(y is None for y in ys):
            print("WARN: %s has missing tier data -- skipping" % display_name)
            continue

        linestyle = "--" if display_name in REASONING_MODELS else "-"
        color = HIGHLIGHT_COLORS[idx]
        marker = HIGHLIGHT_MARKERS[idx]

        ax.plot(
            x_pos, ys,
            label=display_name,
            color=color,
            marker=marker,
            markersize=10,
            linewidth=2.5,
            linestyle=linestyle,
            alpha=0.95,
            zorder=5,
        )

        # Data labels at each point
        for xi, yi in zip(x_pos, ys):
            ax.annotate(
                "%.1f%%" % yi,
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(TIERS, fontsize=12)
    ax.set_xlabel("Question Complexity Tier", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Compositional Drop-off: Basic -> Compound -> Detailed", fontsize=14)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)

    # Legend inside, upper right, not overlapping data
    ax.legend(
        loc="upper right",
        fontsize=9.5,
        framealpha=0.9,
    )

    fig.tight_layout()

    pdf_path = os.path.join(OUTPUT_DIR, "compositional_dropoff.pdf")
    png_path = os.path.join(OUTPUT_DIR, "compositional_dropoff.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print("Saved: %s" % os.path.abspath(pdf_path))
    print("Saved: %s" % os.path.abspath(png_path))


if __name__ == "__main__":
    main()
