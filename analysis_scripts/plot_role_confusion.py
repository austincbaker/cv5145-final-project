#!/usr/bin/env python3
"""
Generate visual confusion matrix heatmaps for role identification.

Produces one PNG per model showing a color-coded confusion matrix with
percentages and counts in each cell.

Usage:
    python analysis_scripts/plot_role_confusion.py -o analysis_scripts/output/role_confusion_plots/
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path


MODEL_FILE_MAP = {
    "InternVL2.5-78B-AWQ": "InternVL2.5-78B-AWQ_combined.json",
    "Qwen3-VL-8B": "qwen3_8B_combined.json",
    "Ovis2.5-9B": "Ovis2.5-9B_combined.json",
    "gemma-4-26B": "gemma_combined.json",
    "InternVL3-9B": "InternVL3-9B_combined.json",
}

ROLE_QTYPES = {
    "aggressor_identification": "Aggressor",
    "victim_recognition": "Victim",
}

PREDICTED_LABELS = ["Aggressor", "Victim", "Bystander", "Unknown"]
PREDICTED_KEYS = ["aggressor", "victim", "bystander", "unknown/cross-video"]


def _slot_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_tok = set(a.lower().split()) - {"a", "the", "in", "with", "and", "person", "wearing"}
    b_tok = set(b.lower().split()) - {"a", "the", "in", "with", "and", "person", "wearing"}
    if not a_tok or not b_tok:
        return False
    return len(a_tok & b_tok) / max(1, len(a_tok | b_tok)) >= 0.4


def classify_selection(selected_text: str, ann: dict) -> str:
    agg = str(ann.get("aggressor") or "")
    vic = str(ann.get("victim") or "")
    bys = ann.get("bystanders") or ""
    if _slot_match(selected_text, agg):
        return "aggressor"
    if _slot_match(selected_text, vic):
        return "victim"
    bys_list = []
    if isinstance(bys, list):
        bys_list = [str(b) for b in bys if b]
    elif isinstance(bys, str) and bys.strip():
        bys_list = [b.strip() for b in bys.replace(" and ", ",").split(",") if b.strip()]
    for b in bys_list:
        if _slot_match(selected_text, b):
            return "bystander"
    return "unknown/cross-video"


def build_matrix(data: dict, ann_map: dict) -> np.ndarray:
    true_labels = ["Aggressor", "Victim"]
    matrix = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for r in data["results"]:
        if r.get("error"):
            continue
        qtype = r["question_type"]
        if qtype not in ROLE_QTYPES:
            continue
        true_role = ROLE_QTYPES[qtype]
        totals[true_role] += 1

        if r["is_correct"]:
            matrix[true_role][true_role] += 1
            continue

        selected_idx = r.get("model_selected_index")
        answers = r.get("answers", [])
        if selected_idx is None or selected_idx < 0 or selected_idx >= len(answers):
            matrix[true_role]["Unknown"] += 1
            continue

        ann = ann_map.get(r["video_name"])
        if not ann:
            matrix[true_role]["Unknown"] += 1
            continue

        predicted_key = classify_selection(answers[selected_idx], ann)
        key_to_label = dict(zip(PREDICTED_KEYS, PREDICTED_LABELS))
        matrix[true_role][key_to_label[predicted_key]] += 1

    arr = np.zeros((len(true_labels), len(PREDICTED_LABELS)))
    for i, tr in enumerate(true_labels):
        t = max(1, totals[tr])
        for j, pr in enumerate(PREDICTED_LABELS):
            arr[i, j] = matrix[tr][pr] / t * 100

    counts = np.zeros((len(true_labels), len(PREDICTED_LABELS)), dtype=int)
    for i, tr in enumerate(true_labels):
        for j, pr in enumerate(PREDICTED_LABELS):
            counts[i, j] = matrix[tr][pr]

    return arr, counts, totals


def plot_matrix(arr: np.ndarray, counts: np.ndarray, totals: dict,
                model_name: str, output_path: Path):
    true_labels = ["Aggressor", "Victim"]
    fig, ax = plt.subplots(figsize=(8, 4))

    annot = np.empty_like(arr, dtype=object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            annot[i, j] = f"{arr[i, j]:.1f}%\n({counts[i, j]})"

    sns.heatmap(arr, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
                xticklabels=PREDICTED_LABELS, yticklabels=true_labels,
                linewidths=1, linecolor="white",
                cbar_kws={"label": "Percentage (%)"},
                ax=ax)

    ax.set_xlabel("Predicted Role", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Role", fontsize=12, fontweight="bold")
    ax.set_title(f"Role Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")

    agg_total = totals.get("Aggressor", 0)
    vic_total = totals.get("Victim", 0)
    ax.set_yticklabels([f"Aggressor\n(n={agg_total})", f"Victim\n(n={vic_total})"],
                       rotation=0, fontsize=10)
    ax.set_xticklabels(PREDICTED_LABELS, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--results-dir", default="combined_results/")
    parser.add_argument("-o", "--output-dir", default="analysis_scripts/output/role_confusion_plots")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, encoding="utf-8") as f:
        anns = json.load(f)
    ann_map = {a["file_name"]: a for a in anns}

    results_dir = Path(args.results_dir)

    for name, filename in MODEL_FILE_MAP.items():
        path = results_dir / filename
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        arr, counts, totals = build_matrix(data, ann_map)
        safe_name = name.replace("/", "_").replace(" ", "_")
        plot_matrix(arr, counts, totals, name, out / f"role_confusion_{safe_name}.png")

    # Combined figure with all models
    models_with_data = []
    for name, filename in MODEL_FILE_MAP.items():
        path = results_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        arr, counts, totals = build_matrix(data, ann_map)
        models_with_data.append((name, arr, counts, totals))

    if models_with_data:
        n = len(models_with_data)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        if n == 1:
            axes = [axes]

        true_labels = ["Aggressor", "Victim"]
        for idx, (name, arr, counts, totals) in enumerate(models_with_data):
            ax = axes[idx]
            annot = np.empty_like(arr, dtype=object)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    annot[i, j] = f"{arr[i, j]:.1f}%\n({counts[i, j]})"

            sns.heatmap(arr, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
                        xticklabels=PREDICTED_LABELS,
                        yticklabels=[f"Agg\n(n={totals.get('Aggressor',0)})",
                                     f"Vic\n(n={totals.get('Victim',0)})"],
                        linewidths=1, linecolor="white",
                        cbar=idx == n - 1,
                        cbar_kws={"label": "%"} if idx == n - 1 else {},
                        ax=ax)
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted" if idx == n // 2 else "", fontsize=9)
            if idx == 0:
                ax.set_ylabel("True Role", fontsize=10)

        plt.suptitle("Role Confusion Matrices - Top 5 Models", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(out / "role_confusion_all_models.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out / 'role_confusion_all_models.png'}")


if __name__ == "__main__":
    main()
