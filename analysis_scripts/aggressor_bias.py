"""Show how models are biased toward assigning the aggressor role.

For role_identification questions, computes:
  - Vic->Agg confusion: when true role is Victim, how often model says Aggressor
  - Agg->Vic confusion: when true role is Aggressor, how often model says Victim
  - Aggressor prediction rate: fraction of all predictions that are "Aggressor"

The asymmetry (Vic->Agg >> Agg->Vic) reveals systematic aggressor bias.
"""
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

ROLE_ORDER = ["Aggressor", "Victim", "Bystander", "No one in the video fits that description"]


def clean_role(label):
    s = label.strip()
    if len(s) >= 3 and s[0].isalpha() and s[1] == ")" and s[2] == " ":
        s = s[3:].strip()
    return s


def match_role(label):
    s = clean_role(label).lower()
    for role in ROLE_ORDER:
        if role.lower() in s or s in role.lower():
            return role
    return None


os.makedirs(OUTPUT_DIR, exist_ok=True)

model_stats = {}

for filename, display_name in MODEL_FILES.items():
    path = os.path.join(COMBINED_DIR, filename)
    if not os.path.exists(path):
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pred_counts = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    total_role = 0

    for r in data["results"]:
        if r["question_type"] != "role_identification":
            continue

        ci = r.get("correct_index")
        mi = r.get("model_selected_index")
        if ci is None or mi is None or mi == -1:
            continue

        answers = r.get("answers") or r.get("all_answers") or []
        if ci >= len(answers) or mi >= len(answers):
            continue

        true_role = match_role(answers[ci])
        pred_role = match_role(answers[mi])
        if true_role is None or pred_role is None:
            continue

        total_role += 1
        pred_counts[pred_role] += 1
        confusion[true_role][pred_role] += 1

    if total_role == 0:
        continue

    agg_true = sum(confusion["Aggressor"].values())
    vic_true = sum(confusion["Victim"].values())

    vic_to_agg = 100.0 * confusion["Victim"]["Aggressor"] / max(vic_true, 1)
    agg_to_vic = 100.0 * confusion["Aggressor"]["Victim"] / max(agg_true, 1)
    byst_to_agg = 100.0 * confusion["Bystander"]["Aggressor"] / max(sum(confusion["Bystander"].values()), 1)
    agg_pred_rate = 100.0 * pred_counts["Aggressor"] / total_role

    model_stats[display_name] = {
        "vic_to_agg": vic_to_agg,
        "agg_to_vic": agg_to_vic,
        "byst_to_agg": byst_to_agg,
        "agg_pred_rate": agg_pred_rate,
        "bias_ratio": vic_to_agg / max(agg_to_vic, 0.1),
    }

    print("%-22s Vic->Agg: %5.1f%%  Agg->Vic: %5.1f%%  Byst->Agg: %5.1f%%  Agg pred rate: %5.1f%%  Bias: %.1fx" % (
        display_name, vic_to_agg, agg_to_vic, byst_to_agg, agg_pred_rate,
        vic_to_agg / max(agg_to_vic, 0.1)))

vic_to_agg_vals = [model_stats[m]["vic_to_agg"] for m in model_stats]
agg_to_vic_vals = [model_stats[m]["agg_to_vic"] for m in model_stats]
byst_to_agg_vals = [model_stats[m]["byst_to_agg"] for m in model_stats]

categories = ["Victim -> Aggressor", "Bystander -> Aggressor", "Aggressor -> Victim"]
data = [vic_to_agg_vals, byst_to_agg_vals, agg_to_vic_vals]
colors = ["#d62728", "#ff7f0e", "#1f77b4"]

fig, ax = plt.subplots(figsize=(7, 5))

bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.5,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

np.random.seed(42)
for i, (vals, color) in enumerate(zip(data, colors)):
    jitter = np.random.normal(0, 0.06, len(vals))
    ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
               color=color, alpha=0.7, s=40, zorder=3, edgecolors="white", linewidth=0.5)

ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel("Confusion Rate (%)", fontsize=11)
ax.set_title("Aggressor Role Bias in VLMs", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

v2a_med = np.median(vic_to_agg_vals)
a2v_med = np.median(agg_to_vic_vals)
b2a_med = np.median(byst_to_agg_vals)
v2a_max = max(vic_to_agg_vals)
v2a_max_model = [m for m in model_stats if model_stats[m]["vic_to_agg"] == v2a_max][0]
n_biased = sum(1 for m in model_stats if model_stats[m]["vic_to_agg"] > model_stats[m]["agg_to_vic"])

desc = (
    "Median Vic->Agg: %.1f%%  |  Median Agg->Vic: %.1f%%  (%.1fx asymmetry)\n"
    "%d/%d models show aggressor bias  |  Worst: %s (%.1f%%)"
    % (v2a_med, a2v_med, v2a_med / max(a2v_med, 0.1),
       n_biased, len(model_stats), v2a_max_model, v2a_max)
)
ax.text(0.5, -0.18, desc, transform=ax.transAxes, fontsize=9,
        ha="center", va="top", style="italic", color="#444444")

plt.tight_layout(rect=[0, 0.08, 1, 1])
out = os.path.join(OUTPUT_DIR, "aggressor_bias")
plt.savefig(out + ".pdf", bbox_inches="tight", dpi=150)
plt.savefig(out + ".png", bbox_inches="tight", dpi=150)
plt.close()
print("\nSaved %s" % out)
