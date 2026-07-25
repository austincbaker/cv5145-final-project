#!/usr/bin/env python3
"""Generate a heatmap of accuracy for each (model, question_type) pair."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "combined_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Model files -> display names
# General-purpose models
GENERAL_PURPOSE_MODELS = [
    ("gemma_combined.json", "Gemma-4-26B"),
    ("InternVideo2_5_Chat_8B_combined.json", "InternVideo2.5-8B"),
    ("InternVL2.5-8B_combined.json", "InternVL2.5-8B"),
    ("InternVL2.5-78B-AWQ_combined.json", "InternVL2.5-78B"),
    ("InternVL3-9B_combined.json", "InternVL3-9B"),
    ("internvl3_5_combined.json", "InternVL3.5-8B"),
    ("LLaVA-Video-7B-Qwen2_combined.json", "LLaVA-Video-7B"),
    ("Ovis2.5-9B_combined.json", "Ovis2.5-9B"),
    ("qwen2_5_7B_combined.json", "Qwen2.5-VL-7B"),
    ("qwen2_5_72B_combined.json", "Qwen2.5-VL-72B"),
    ("qwen3_8B_combined.json", "Qwen3-VL-8B"),
    ("VideoLLaMA3-7B_combined.json", "VideoLLaMA3-7B"),
]

# Reasoning variants
REASONING_MODELS = [
    ("Ovis2.5-9B-Thinking_combined.json", "Ovis2.5-9B-Think"),
    ("qwen3_8B_thinking_combined.json", "Qwen3-VL-8B-Think"),
    ("internvl3_5_cot_combined.json", "InternVL3.5-8B-CoT"),
    ("internvl3_5_dot_combined.json", "InternVL3.5-8B-DoT"),
]

# Question types -> display names, grouped by tier
# Order: Basic, Compound, Detailed (sequence), Secondary
QUESTION_TYPE_GROUPS = {
    "Basic": [
        ("primary_action", "Action"),
        ("aggressor_identification", "Aggressor ID"),
        ("victim_recognition", "Victim ID"),
        ("role_identification", "Role ID"),
    ],
    "Compound": [
        ("compound_action_aggressor", "Act+Agg"),
        ("compound_action_victims", "Act+Vic"),
        ("compound_aggressor_victim", "Agg+Vic"),
        ("compound_aggressor_action_victim", "Agg+Act+Vic"),
    ],
    "Detailed": [
        ("sequence_verification", "Sequence"),
    ],
    "Secondary": [
        ("compound_action_location", "Act+Loc"),
        ("compound_aggressor_victim_count", "Agg+Vic Count"),
        ("role_count_aggressor", "Count Agg"),
        ("role_count_victim", "Count Vic"),
        ("role_count_bystander", "Count Byst"),
    ],
}

# Flatten for ordered column list
ALL_QTYPES = []
QTYPE_DISPLAY = {}
GROUP_BOUNDARIES = []  # indices where a new group starts
for group_name, items in QUESTION_TYPE_GROUPS.items():
    GROUP_BOUNDARIES.append(len(ALL_QTYPES))
    for raw, display in items:
        ALL_QTYPES.append(raw)
        QTYPE_DISPLAY[raw] = display


def load_model_data(filename):
    """Load results from a model JSON file and compute per-question-type accuracy."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print("WARN: File not found: %s" % str(filepath).encode("ascii", "replace").decode())
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for entry in results:
        qt = entry.get("question_type", "")
        is_correct = entry.get("is_correct", False)
        counts[qt]["total"] += 1
        if is_correct:
            counts[qt]["correct"] += 1

    accuracies = {}
    overall_correct = 0
    overall_total = 0
    for qt in ALL_QTYPES:
        c = counts.get(qt)
        if c and c["total"] > 0:
            accuracies[qt] = c["correct"] / c["total"] * 100.0
            overall_correct += c["correct"]
            overall_total += c["total"]
        else:
            accuracies[qt] = float("nan")

    # Compute overall accuracy across all question types
    if overall_total > 0:
        accuracies["__overall__"] = overall_correct / overall_total * 100.0
    else:
        accuracies["__overall__"] = float("nan")

    return accuracies


def build_heatmap_matrix():
    """Build the 2D matrix (models x question_types + Overall) of accuracy values.

    Models are sorted by overall accuracy (best at top) within each group
    (general-purpose first, reasoning variants second).
    An "Average" row is appended at the bottom.
    """
    # Load data for each group separately
    def load_group(model_list):
        entries = []
        for filename, display_name in model_list:
            accs = load_model_data(filename)
            if accs is None:
                continue
            row = [accs[qt] for qt in ALL_QTYPES] + [accs["__overall__"]]
            entries.append((display_name, row))
        # Sort by overall accuracy descending (last element in row)
        entries.sort(key=lambda x: x[1][-1] if not np.isnan(x[1][-1]) else -1, reverse=True)
        return entries

    general_entries = load_group(GENERAL_PURPOSE_MODELS)
    reasoning_entries = load_group(REASONING_MODELS)

    model_names = []
    matrix = []

    for name, row in general_entries:
        model_names.append(name)
        matrix.append(row)

    reasoning_start_idx = len(model_names)

    for name, row in reasoning_entries:
        model_names.append(name)
        matrix.append(row)

    # Compute "Average" row across all models for each column
    if matrix:
        data_arr = np.array(matrix)
        avg_row = []
        for col_idx in range(data_arr.shape[1]):
            col_vals = data_arr[:, col_idx]
            valid = col_vals[~np.isnan(col_vals)]
            if len(valid) > 0:
                avg_row.append(np.mean(valid))
            else:
                avg_row.append(float("nan"))
        model_names.append("Average")
        matrix.append(avg_row)

    return model_names, matrix, reasoning_start_idx


def print_markdown_table(model_names, matrix):
    """Print a markdown table of the accuracy data to stdout (ASCII-safe)."""
    col_labels = [QTYPE_DISPLAY[qt] for qt in ALL_QTYPES] + ["Overall"]

    # Header
    header = "| Model | " + " | ".join(col_labels) + " |"
    sep = "|---|" + "|".join(["---"] * len(col_labels)) + "|"
    print(header.encode("ascii", "replace").decode())
    print(sep)

    for name, row in zip(model_names, matrix):
        cells = []
        for val in row:
            if np.isnan(val):
                cells.append("--")
            else:
                cells.append("%.1f" % val)
        line = "| %s | %s |" % (name, " | ".join(cells))
        print(line.encode("ascii", "replace").decode())


def create_heatmap(model_names, matrix, reasoning_start_idx):
    """Create and save the heatmap figure."""
    col_labels = [QTYPE_DISPLAY[qt] for qt in ALL_QTYPES] + ["Overall"]
    data = np.array(matrix)

    fig, ax = plt.subplots(figsize=(18, 10))

    # Diverging colormap centered around 50% (chance level)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)

    # Draw heatmap
    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        xticklabels=col_labels,
        yticklabels=model_names,
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
        annot_kws={"size": 8},
    )

    # Add vertical separator lines between tier groups
    for boundary_idx in GROUP_BOUNDARIES[1:]:
        ax.axvline(x=boundary_idx, color="black", linewidth=2.5)

    # Add vertical separator before "Overall" column
    overall_col_idx = len(ALL_QTYPES)
    ax.axvline(x=overall_col_idx, color="black", linewidth=2.5)

    # Add horizontal separator between general-purpose and reasoning variants
    if reasoning_start_idx is not None and reasoning_start_idx < len(model_names):
        ax.axhline(y=reasoning_start_idx, color="black", linewidth=2.5)

    # Add horizontal separator before "Average" row (last row)
    avg_row_idx = len(model_names) - 1
    ax.axhline(y=avg_row_idx, color="black", linewidth=2.5)

    # Add tier group labels at the top
    group_names = list(QUESTION_TYPE_GROUPS.keys())
    group_sizes = [len(v) for v in QUESTION_TYPE_GROUPS.values()]
    cumulative = 0
    for gname, gsize in zip(group_names, group_sizes):
        center = cumulative + gsize / 2.0
        ax.text(
            center, -0.8, gname,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        cumulative += gsize

    ax.set_title(
        "Model Accuracy by Question Type", fontsize=16, fontweight="bold", pad=30
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Rotate x labels
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "question_type_heatmap.pdf"
    png_path = OUTPUT_DIR / "question_type_heatmap.png"
    fig.savefig(str(pdf_path), bbox_inches="tight", dpi=150)
    fig.savefig(str(png_path), bbox_inches="tight", dpi=150)
    plt.close(fig)

    print("Saved: %s" % str(pdf_path))
    print("Saved: %s" % str(png_path))


def main():
    print("Loading model results from: %s" % str(RESULTS_DIR))
    model_names, matrix, reasoning_start_idx = build_heatmap_matrix()
    # Subtract 1 for Average row in the count
    n_models = len(model_names) - 1 if model_names and model_names[-1] == "Average" else len(model_names)
    print("Loaded %d models, %d question types" % (n_models, len(ALL_QTYPES)))
    print("")

    print_markdown_table(model_names, matrix)
    print("")

    create_heatmap(model_names, matrix, reasoning_start_idx)
    print("Done.")


if __name__ == "__main__":
    main()
