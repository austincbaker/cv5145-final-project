#!/usr/bin/env python3
"""Analyze per-model accuracy on trick (negation) vs non-trick questions.

Joins eval results to the question bank by (video_name, prompt) to get
the authoritative is_trick flag, then computes accuracy breakdowns and
generates a grouped bar chart for 5 selected models with a gray band
showing the range across all 16 models.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "combined_results")
QUESTION_BANK = os.path.join(BASE_DIR, "train_model", "data", "generated_questions.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_scripts", "output")

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

# 5 selected models to plot bars for
SELECTED_MODEL_FILES = {
    "internvl3_5_combined.json": "InternVL3.5-8B",
    "gemma_combined.json": "Gemma-4-26B",
    "qwen2_5_72B_combined.json": "Qwen2.5-VL-72B",
    "LLaVA-Video-7B-Qwen2_combined.json": "LLaVA-Video-7B",
    "qwen3_8B_combined.json": "Qwen3-VL-8B",
}

# ---------------------------------------------------------------------------
# Load question bank -> (video_name, prompt) -> is_trick
# ---------------------------------------------------------------------------
print("Loading question bank...")
with open(QUESTION_BANK, encoding="utf-8") as f:
    qbank = json.load(f)

trick_lookup = {}
for vid, qs in qbank["questions_by_video"].items():
    for q in qs:
        key = (q["video_name"], q["prompt"])
        trick_lookup[key] = bool(q.get("is_trick", False))

total_trick = sum(1 for v in trick_lookup.values() if v)
total_nontrick = sum(1 for v in trick_lookup.values() if not v)
print("Question bank: %d trick, %d non-trick, %d total" % (
    total_trick, total_nontrick, len(trick_lookup)))


# ---------------------------------------------------------------------------
# Helper: compute trick/nontrick accuracy for a model file
# ---------------------------------------------------------------------------
def compute_model_stats(fname):
    """Return (trick_acc, nontrick_acc, gap, trick_n, nontrick_n) or None."""
    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        print("WARN: missing %s -- skipping" % fname)
        return None

    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    trick_correct = 0
    trick_total = 0
    nontrick_correct = 0
    nontrick_total = 0
    unmatched = 0

    for r in results:
        key = (r["video_name"], r["prompt"])
        is_trick = trick_lookup.get(key)
        if is_trick is None:
            unmatched += 1
            continue
        correct = bool(r.get("is_correct", False))
        if is_trick:
            trick_total += 1
            if correct:
                trick_correct += 1
        else:
            nontrick_total += 1
            if correct:
                nontrick_correct += 1

    if trick_total == 0 or nontrick_total == 0:
        print("WARN: %s has 0 trick or non-trick questions -- skipping" % fname)
        return None

    trick_acc = trick_correct / trick_total * 100
    nontrick_acc = nontrick_correct / nontrick_total * 100
    gap = nontrick_acc - trick_acc

    if unmatched > 0:
        print("  %s: %d results unmatched in question bank" % (fname, unmatched))

    return (trick_acc, nontrick_acc, gap, trick_total, nontrick_total)


# ---------------------------------------------------------------------------
# Process ALL models (for gray band)
# ---------------------------------------------------------------------------
all_rows = []  # (display_name, trick_acc, nontrick_acc, gap, trick_n, nontrick_n)

for fname, display in sorted(ALL_MODEL_FILES.items(), key=lambda x: x[1]):
    stats = compute_model_stats(fname)
    if stats is None:
        continue
    trick_acc, nontrick_acc, gap, trick_n, nontrick_n = stats
    all_rows.append((display, trick_acc, nontrick_acc, gap, trick_n, nontrick_n))

# Compute gray band range from all models
all_trick_accs = [r[1] for r in all_rows]
all_nontrick_accs = [r[2] for r in all_rows]
trick_min = min(all_trick_accs)
trick_max = max(all_trick_accs)
nontrick_min = min(all_nontrick_accs)
nontrick_max = max(all_nontrick_accs)

print()
print("All 16 models -- trick acc range: %.1f%% -- %.1f%%" % (trick_min, trick_max))
print("All 16 models -- non-trick acc range: %.1f%% -- %.1f%%" % (nontrick_min, nontrick_max))

# ---------------------------------------------------------------------------
# Filter to the 5 selected models
# ---------------------------------------------------------------------------
selected_names = set(SELECTED_MODEL_FILES.values())
rows = [r for r in all_rows if r[0] in selected_names]

# Sort by gap descending (largest gap first)
rows.sort(key=lambda x: x[3], reverse=True)

# ---------------------------------------------------------------------------
# Print markdown table
# ---------------------------------------------------------------------------
print()
print("## Trick vs Non-Trick Accuracy (5 selected models)")
print()
print("| Model | Trick Acc%% | Non-Trick Acc%% | Gap |")
print("|---|---|---|---|")
for display, trick_acc, nontrick_acc, gap, trick_n, nontrick_n in rows:
    print("| %s | %.1f | %.1f | %.1f |" % (display, trick_acc, nontrick_acc, gap))

print()
print("Question counts -- trick: %d, non-trick: %d" % (total_trick, total_nontrick))

# ---------------------------------------------------------------------------
# Grouped bar chart -- 5 models, with gray band for all-model range
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 7))

labels = [r[0] for r in rows]
trick_accs = [r[1] for r in rows]
nontrick_accs = [r[2] for r in rows]
gaps = [r[3] for r in rows]

x = np.arange(len(labels))
width = 0.32

# Gray shaded region showing the range of all 16 models' trick accuracy
# Spans the full x-axis behind the bars
ax.axhspan(trick_min, trick_max, color="#DDDDDD", alpha=0.5, zorder=0,
           label="All-model trick acc range")

# Bars -- thicker (width=0.32) with stronger edges
bars_nontrick = ax.bar(x - width / 2, nontrick_accs, width, label="Non-Negation",
                        color="#4C72B0", edgecolor="white", linewidth=0.8, zorder=2)
bars_trick = ax.bar(x + width / 2, trick_accs, width, label="Negation",
                     color="#DD8452", edgecolor="white", linewidth=0.8, zorder=2)

# Gap annotations with color coding
for i, gap_val in enumerate(gaps):
    max_height = max(nontrick_accs[i], trick_accs[i])
    # Color: red if gap > 20pp, orange if 10-20pp, green if < 10pp
    abs_gap = abs(gap_val)
    if abs_gap > 20:
        gap_color = "#C44E52"  # red
    elif abs_gap > 10:
        gap_color = "#E08B2D"  # orange
    else:
        gap_color = "#55A868"  # green
    ax.annotate(
        "%+.1f pp" % gap_val,
        xy=(x[i], max_height + 1.5),
        ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        color=gap_color,
        zorder=3,
    )

# Chance baseline at 12.5% for 8-option questions
ax.axhline(y=12.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)
ax.text(len(labels) - 0.5, 13.5, "chance (12.5%%, 8 options)", fontsize=8, color="gray",
        ha="right", va="bottom")

ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Model Accuracy: Negation vs Non-Negation Questions\n"
             "(5 representative models)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
ax.set_ylim(0, max(max(nontrick_accs), max(trick_accs)) + 15)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)

plt.tight_layout()

pdf_path = os.path.join(OUTPUT_DIR, "trick_question_analysis.pdf")
png_path = os.path.join(OUTPUT_DIR, "trick_question_analysis.png")
fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print()
print("Saved: %s" % pdf_path)
print("Saved: %s" % png_path)
