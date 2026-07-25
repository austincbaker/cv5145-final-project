"""Show how distractor vulnerability shifts across tiers (Basic -> Compound -> Detailed).

Grouped bar chart: distractor types on x-axis, bars grouped by tier, values averaged across models.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_DIR = os.path.join(BASE_DIR, "combined_results")
QUESTION_BANK = os.path.join(BASE_DIR, "train_model", "data", "generated_questions.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_scripts", "output")

MODEL_FILES = {
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

DISTRACTOR_TYPES = [
    "role_reversal", "cross_video", "wrong_action", "wrong_category",
    "wrong_aggressor", "wrong_victim", "bystander_substitution",
    "wrong_location", "frequency_saturation", "none_claim",
]

DISTRACTOR_LABELS = [
    "Role\nReversal", "Cross\nVideo", "Wrong\nAction", "Wrong\nCategory",
    "Wrong\nAggressor", "Wrong\nVictim", "Bystander\nSubstitution",
    "Wrong\nLocation", "Frequency\nSaturation", "None\nClaim",
]

TIERS = {
    "Basic": ["primary_action", "role_identification", "aggressor_identification", "victim_recognition"],
    "Compound": ["compound_action_aggressor", "compound_action_victims", "compound_aggressor_victim"],
    "Detailed": ["compound_aggressor_action_victim", "sequence_verification"],
}

print("Loading question bank...")
with open(QUESTION_BANK, encoding="utf-8") as f:
    qb = json.load(f)

q_lookup = {}
q_type_lookup = {}
for video_name, questions in qb["questions_by_video"].items():
    for q in questions:
        key = (q["video_name"], q["prompt"])
        q_lookup[key] = q["option_hardness"]
        q_type_lookup[key] = q["question_type"]

# tier -> distractor_type -> list of per-model percentages
tier_distractor_pcts = {t: defaultdict(list) for t in TIERS}

for filename, display_name in MODEL_FILES.items():
    path = os.path.join(COMBINED_DIR, filename)
    if not os.path.exists(path):
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tier_counts = {t: defaultdict(int) for t in TIERS}
    tier_totals = {t: 0 for t in TIERS}

    for r in data["results"]:
        if r.get("is_correct", False):
            continue
        idx = r.get("model_selected_index")
        if idx is None or idx == -1:
            continue
        key = (r["video_name"], r["prompt"])
        option_hardness = q_lookup.get(key)
        qt = q_type_lookup.get(key, r.get("question_type", ""))
        if option_hardness is None or idx < 0 or idx >= len(option_hardness):
            continue
        hardness = option_hardness[idx]
        if hardness == "correct":
            continue

        for tier_name, tier_types in TIERS.items():
            if qt in tier_types:
                tier_counts[tier_name][hardness] += 1
                tier_totals[tier_name] += 1
                break

    for tier_name in TIERS:
        t = max(tier_totals[tier_name], 1)
        for dt in DISTRACTOR_TYPES:
            tier_distractor_pcts[tier_name][dt].append(100.0 * tier_counts[tier_name][dt] / t)

# Compute means and stds
tier_names = ["Basic", "Compound", "Detailed"]
colors = ["#4c78a8", "#f58518", "#e45756"]

means = {t: [np.mean(tier_distractor_pcts[t][dt]) for dt in DISTRACTOR_TYPES] for t in tier_names}
stds = {t: [np.std(tier_distractor_pcts[t][dt]) for dt in DISTRACTOR_TYPES] for t in tier_names}

x = np.arange(len(DISTRACTOR_TYPES))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))

for i, (tier, color) in enumerate(zip(tier_names, colors)):
    bars = ax.bar(x + i * width, means[tier], width, yerr=stds[tier],
                  label=tier, color=color, alpha=0.85, capsize=3, error_kw={"linewidth": 0.8})
    for bar, val in zip(bars, means[tier]):
        if val > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    "%.1f" % val, ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(DISTRACTOR_LABELS, fontsize=9, ha="center")
ax.set_ylabel("Mean % of Errors (across models)", fontsize=11)
ax.set_title("Distractor Vulnerability Shift Across Question Tiers", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.set_ylim(0, 85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(y=12.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(len(DISTRACTOR_TYPES) - 0.5, 13.5, "chance (12.5%)", fontsize=7, color="gray", ha="right")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "distractor_shift_by_tier")
plt.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved %s" % out)
