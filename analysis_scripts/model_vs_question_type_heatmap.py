"""Heatmap of model accuracy per question type (models x question categories)."""
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_DIR = os.path.join(BASE_DIR, "combined_results")
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

QUESTION_TYPE_ORDER = [
    "primary_action", "role_identification", "aggressor_identification", "victim_recognition",
    "compound_action_aggressor", "compound_action_victims", "compound_aggressor_victim",
    "compound_aggressor_action_victim", "sequence_verification",
    "compound_action_location", "role_count_victim", "role_count_aggressor",
    "role_count_bystander", "compound_aggressor_victim_count",
]

QUESTION_TYPE_LABELS = [
    "Action", "Role ID", "Aggressor ID", "Victim ID",
    "Act+Agg", "Act+Vic", "Agg+Vic",
    "Agg+Act+Vic", "Seq Verify",
    "Act+Loc", "Count Vic", "Count Agg",
    "Count Byst", "Agg+Vic Count",
]

TIER_BOUNDARIES = [4, 7, 9]

model_data = {}

for filename, display_name in MODEL_FILES.items():
    path = os.path.join(COMBINED_DIR, filename)
    if not os.path.exists(path):
        print("SKIP %s" % filename)
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in data["results"]:
        qt = r["question_type"]
        by_type[qt]["total"] += 1
        if r.get("is_correct", False):
            by_type[qt]["correct"] += 1
        elif "model_response" in r and "model_selected_index" not in r:
            resp = r.get("model_response", "").strip()
            m = re.search(r"(\d+)", resp)
            if m and int(m.group(1)) - 1 == r["correct_index"]:
                by_type[qt]["correct"] += 1

    row = []
    for qt in QUESTION_TYPE_ORDER:
        d = by_type[qt]
        row.append(100.0 * d["correct"] / max(d["total"], 1) if d["total"] > 0 else float("nan"))
    model_data[display_name] = row

models = sorted(model_data.keys(), key=lambda m: np.nanmean(model_data[m]), reverse=True)
matrix = np.array([model_data[m] for m in models])

fig, ax = plt.subplots(figsize=(14, max(6, len(models) * 0.42)))
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=10, vmax=70)

ax.set_xticks(range(len(QUESTION_TYPE_LABELS)))
ax.set_xticklabels(QUESTION_TYPE_LABELS, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)

for i in range(len(models)):
    for j in range(len(QUESTION_TYPE_LABELS)):
        val = matrix[i, j]
        if not np.isnan(val):
            color = "white" if val < 20 or val > 65 else "black"
            ax.text(j, i, "%.1f" % val, ha="center", va="center", fontsize=7.5, fontweight="bold", color=color)

for b in TIER_BOUNDARIES:
    ax.axvline(x=b - 0.5, color="white", linewidth=2)

ax.set_title("Model Accuracy by Question Category (%)", fontsize=13, fontweight="bold", pad=12)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Accuracy (%)", fontsize=10)

ax.text(1.5, -1.8, "Basic", ha="center", fontsize=10, fontweight="bold")
ax.text(5.5, -1.8, "Compound", ha="center", fontsize=10, fontweight="bold")
ax.text(7.5, -1.8, "Detailed", ha="center", fontsize=10, fontweight="bold")
ax.text(11.5, -1.8, "Secondary", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "model_vs_question_type_heatmap")
plt.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved %s" % out)
