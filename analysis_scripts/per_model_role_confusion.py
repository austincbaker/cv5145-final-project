#!/usr/bin/env python3
"""Generate per-model role confusion matrices for role_identification questions.

Reads combined eval results, filters to role_identification, builds a confusion
matrix (true role vs predicted role) for each model, and plots a grid of heatmaps.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Model file -> display name mapping
# ---------------------------------------------------------------------------
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

# Canonical role order for the confusion matrix
ROLE_ORDER = ["Aggressor", "Victim", "Bystander", "No one in the video fits that description"]
ROLE_SHORT = ["Aggressor", "Victim", "Bystander", "Not in video"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "combined_results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def clean_role_label(label):
    """Strip any A)/B)/etc prefix and whitespace from a role label."""
    s = label.strip()
    # Handle "A) Aggressor" style if present
    if len(s) >= 3 and s[0].isalpha() and s[1] == ")" and s[2] == " ":
        s = s[3:].strip()
    return s


def build_confusion_matrix(results):
    """Build a raw count confusion matrix from role_identification results.

    Returns (matrix, n_skipped) where matrix is len(ROLE_ORDER) x len(ROLE_ORDER).
    Rows = true role, Columns = predicted role.
    """
    role_to_idx = {r: i for i, r in enumerate(ROLE_ORDER)}
    n = len(ROLE_ORDER)
    matrix = np.zeros((n, n), dtype=int)
    skipped = 0

    for r in results:
        if r["question_type"] != "role_identification":
            continue

        ci = r.get("correct_index")
        mi = r.get("model_selected_index")

        # Skip invalid indices
        if ci is None or mi is None or mi == -1:
            skipped += 1
            continue
        if ci < 0 or ci >= len(r["answers"]):
            skipped += 1
            continue
        if mi < 0 or mi >= len(r["answers"]):
            skipped += 1
            continue

        true_label = clean_role_label(r["answers"][ci])
        pred_label = clean_role_label(r["answers"][mi])

        if true_label not in role_to_idx or pred_label not in role_to_idx:
            skipped += 1
            continue

        matrix[role_to_idx[true_label], role_to_idx[pred_label]] += 1

    return matrix, skipped


def normalize_rows(matrix):
    """Normalize each row to percentages. Returns percentage matrix."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return (matrix / row_sums) * 100.0


def compute_role_accuracy(results):
    """Compute overall role_identification accuracy (% correct)."""
    correct = 0
    total = 0
    for r in results:
        if r["question_type"] != "role_identification":
            continue
        ci = r.get("correct_index")
        mi = r.get("model_selected_index")
        if ci is None or mi is None or mi == -1:
            continue
        total += 1
        if ci == mi:
            correct += 1
    if total == 0:
        return 0.0
    return (correct / total) * 100.0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all models -- preserve insertion order from MODEL_FILES
    model_data = {}  # display_name -> (raw_matrix, pct_matrix, n_skipped, n_total, accuracy)

    # Use MODEL_FILES insertion order (not alphabetical)
    for fname, display_name in MODEL_FILES.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(fpath):
            print("WARN: File not found, skipping: %s" % fname)
            continue

        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        role_results = [r for r in results if r["question_type"] == "role_identification"]
        matrix, skipped = build_confusion_matrix(results)
        pct_matrix = normalize_rows(matrix)
        accuracy = compute_role_accuracy(results)

        model_data[display_name] = (matrix, pct_matrix, skipped, len(role_results), accuracy)
        total_counted = matrix.sum()
        print(
            "%s: %d role questions, %d valid, %d skipped, accuracy=%.1f%%"
            % (display_name, len(role_results), total_counted, skipped, accuracy)
        )

    if not model_data:
        print("FAIL: No model data loaded.")
        sys.exit(1)

    n_models = len(model_data)
    ncols = 5
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
    axes = np.atleast_2d(axes)

    # Consistent color scale: 0-100%
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.Blues

    # Preserve MODEL_FILES insertion order
    model_names_ordered = list(model_data.keys())

    confusion_summary = []  # (model, agg->vic%, vic->agg%)

    for idx, model_name in enumerate(model_names_ordered):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax = axes[row_idx, col_idx]

        raw_matrix, pct_matrix, _, n_total, accuracy = model_data[model_name]

        im = ax.imshow(pct_matrix, cmap=cmap, norm=norm, aspect="auto")

        # Annotate cells with percentage and raw count
        for i in range(len(ROLE_ORDER)):
            for j in range(len(ROLE_ORDER)):
                pct_val = pct_matrix[i, j]
                raw_val = raw_matrix[i, j]
                # White text on dark cells, black on light
                color = "white" if pct_val > 50 else "black"
                # Thicker border on diagonal cells (correct predictions)
                if i == j:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=2.5,
                    ))
                ax.text(
                    j, i,
                    "%.1f%%\n(%d)" % (pct_val, raw_val),
                    ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold",
                )

        ax.set_xticks(range(len(ROLE_SHORT)))
        ax.set_xticklabels(ROLE_SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ROLE_SHORT)))
        ax.set_yticklabels(ROLE_SHORT, fontsize=8)
        # Title with subtitle showing overall accuracy
        ax.set_title(
            "%s\n(Accuracy: %.1f%%)" % (model_name, accuracy),
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)

        # Track aggressor<->victim confusion
        agg_idx = ROLE_ORDER.index("Aggressor")
        vic_idx = ROLE_ORDER.index("Victim")
        agg_to_vic = pct_matrix[agg_idx, vic_idx]
        vic_to_agg = pct_matrix[vic_idx, agg_idx]
        confusion_summary.append((model_name, agg_to_vic, vic_to_agg))

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row_idx = idx // ncols
        col_idx = idx % ncols
        axes[row_idx, col_idx].set_visible(False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("Row-normalized percentage (%)", fontsize=10)

    fig.suptitle(
        "Role Confusion Matrices by Model (role_identification questions)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.subplots_adjust(
        left=0.05, right=0.91, top=0.85, bottom=0.15,
        hspace=0.45, wspace=0.40,
    )

    # Save
    pdf_path = os.path.join(OUTPUT_DIR, "per_model_role_confusion.pdf")
    png_path = os.path.join(OUTPUT_DIR, "per_model_role_confusion.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print("Saved: %s" % pdf_path)
    print("Saved: %s" % png_path)
    plt.close(fig)

    # Print summary of aggressor<->victim confusion
    print("")
    print("=" * 70)
    print("AGGRESSOR <-> VICTIM CONFUSION SUMMARY")
    print("=" * 70)
    print("")

    # Sort by worst aggressor->victim confusion
    confusion_summary.sort(key=lambda x: x[1], reverse=True)
    print("Worst Aggressor -> Victim confusion (true=Aggressor, predicted=Victim):")
    print("-" * 55)
    for model_name, agg_to_vic, _ in confusion_summary:
        print("  %-22s  %.1f%%" % (model_name, agg_to_vic))

    print("")

    # Sort by worst victim->aggressor confusion
    confusion_summary.sort(key=lambda x: x[2], reverse=True)
    print("Worst Victim -> Aggressor confusion (true=Victim, predicted=Aggressor):")
    print("-" * 55)
    for model_name, _, vic_to_agg in confusion_summary:
        print("  %-22s  %.1f%%" % (model_name, vic_to_agg))


if __name__ == "__main__":
    main()
