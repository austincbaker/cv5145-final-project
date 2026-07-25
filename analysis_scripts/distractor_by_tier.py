"""Distractor vulnerability heatmaps split by question tier (Basic, Compound, Detailed).

For each tier, shows models (y) vs distractor types (x), with cell values as
percentage of errors caused by each distractor type.
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
    "role_reversal", "cross_video", "wrong_action", "wrong_aggressor",
    "wrong_victim", "wrong_category", "wrong_location",
    "bystander_substitution", "frequency_saturation", "none_claim",
]

DISTRACTOR_LABELS = [
    "Role Rev.", "Cross Vid.", "Wrong Act.", "Wrong Agg.",
    "Wrong Vic.", "Wrong Cat.", "Wrong Loc.",
    "Byst. Sub.", "Freq. Sat.", "None Claim",
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
print("  Loaded %d questions" % len(q_lookup))

# tier_name -> model_name -> {distractor_type -> count}
tier_model_counts = {t: {} for t in TIERS}
tier_model_totals = {t: {} for t in TIERS}

for filename, display_name in sorted(MODEL_FILES.items(), key=lambda x: x[1]):
    path = os.path.join(COMBINED_DIR, filename)
    if not os.path.exists(path):
        print("SKIP %s" % filename)
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for tier_name in TIERS:
        tier_model_counts[tier_name][display_name] = defaultdict(int)
        tier_model_totals[tier_name][display_name] = 0

    for r in data["results"]:
        if r.get("is_correct", False):
            continue

        idx = r.get("model_selected_index")
        if idx is None or idx == -1:
            continue

        key = (r["video_name"], r["prompt"])
        option_hardness = q_lookup.get(key)
        qt = q_type_lookup.get(key, r.get("question_type", ""))
        if option_hardness is None:
            continue
        if idx < 0 or idx >= len(option_hardness):
            continue

        hardness = option_hardness[idx]
        if hardness == "correct":
            continue

        for tier_name, tier_types in TIERS.items():
            if qt in tier_types:
                tier_model_counts[tier_name][display_name][hardness] += 1
                tier_model_totals[tier_name][display_name] += 1
                break

    print("  %s done" % display_name)

for tier_name in TIERS:
    counts = tier_model_counts[tier_name]
    totals = tier_model_totals[tier_name]
    models = sorted(counts.keys(), key=lambda m: totals.get(m, 0))

    matrix = np.zeros((len(models), len(DISTRACTOR_TYPES)))
    for i, m in enumerate(models):
        t = max(totals[m], 1)
        for j, dt in enumerate(DISTRACTOR_TYPES):
            matrix[i, j] = 100.0 * counts[m].get(dt, 0) / t

    fig, ax = plt.subplots(figsize=(12, max(5, len(models) * 0.42)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=35)

    ax.set_xticks(range(len(DISTRACTOR_LABELS)))
    ax.set_xticklabels(DISTRACTOR_LABELS, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(DISTRACTOR_TYPES)):
            val = matrix[i, j]
            color = "white" if val > 25 else "black"
            ax.text(j, i, "%.1f" % val, ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    ax.set_title("Distractor Vulnerability -- %s Questions (%%)" % tier_name, fontsize=13, fontweight="bold", pad=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of errors", fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "distractor_by_%s" % tier_name.lower())
    plt.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
    plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved %s" % out)
