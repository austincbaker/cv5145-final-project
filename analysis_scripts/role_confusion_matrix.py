#!/usr/bin/env python3
"""
B3: Role confusion matrix.

For aggressor_identification and victim_recognition questions, when the model
picks the wrong answer, determine what role the selected person actually has
(aggressor, victim, bystander, or unknown/cross-video).

This shows whether models confuse aggressor->victim more than victim->aggressor.

Usage:
    python analysis_scripts/role_confusion_matrix.py -o analysis_scripts/output/csv/
"""
import argparse
import json
import csv
from collections import defaultdict
from pathlib import Path


MODEL_FILE_MAP = {
    "InternVL2.5-78B-AWQ": "InternVL2.5-78B-AWQ_combined.json",
    "Qwen3-VL-8B": "qwen3_8B_combined.json",
    "Ovis2.5-9B": "Ovis2.5-9B_combined.json",
    "gemma-4-26B": "gemma_combined.json",
    "InternVL3-9B": "InternVL3-9B_combined.json",
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--results-dir", default="combined_results/")
    parser.add_argument("--questions", default="train_model/data/generated_questions.json")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("-o", "--output-dir", default="analysis_scripts/output/csv")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, encoding="utf-8") as f:
        anns = json.load(f)
    ann_map = {a["file_name"]: a for a in anns}

    results_dir = Path(args.results_dir)

    overall = []
    for name, filename in MODEL_FILE_MAP.items():
        path = results_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        correct = sum(1 for r in data["results"] if r.get("is_correct"))
        total = sum(1 for r in data["results"] if not r.get("error"))
        overall.append((name, correct / max(1, total) * 100, filename))
    overall.sort(key=lambda x: -x[1])
    top_models = overall[:args.top_n]

    role_qtypes = {
        "aggressor_identification": "aggressor",
        "victim_recognition": "victim",
    }

    predicted_roles = ["aggressor", "victim", "bystander", "unknown/cross-video"]
    true_roles = ["aggressor", "victim"]

    for name, _, filename in top_models:
        path = results_dir / filename
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        matrix = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)

        for r in data["results"]:
            if r.get("error"):
                continue
            qtype = r["question_type"]
            if qtype not in role_qtypes:
                continue

            true_role = role_qtypes[qtype]
            totals[true_role] += 1

            if r["is_correct"]:
                matrix[true_role][true_role] += 1
                continue

            selected_idx = r.get("model_selected_index")
            if selected_idx is None or selected_idx < 0:
                matrix[true_role]["unknown/cross-video"] += 1
                continue
            answers = r.get("answers", [])
            if selected_idx >= len(answers):
                matrix[true_role]["unknown/cross-video"] += 1
                continue

            ann = ann_map.get(r["video_name"])
            if not ann:
                matrix[true_role]["unknown/cross-video"] += 1
                continue

            predicted = classify_selection(answers[selected_idx], ann)
            matrix[true_role][predicted] += 1

        print(f"\n{'='*60}")
        print(f"{name} — Role Confusion Matrix")
        print(f"{'='*60}")
        print(f"  Rows = true role (what the question asked for)")
        print(f"  Cols = what the model selected")
        print()

        label = "True \\ Predicted"
        header = f"  {label:<20}"
        for pr in predicted_roles:
            header += f"  {pr:>12}"
        header += f"  {'Total':>8}"
        print(header)
        print(f"  {'-'*20}" + f"  {'-'*12}" * len(predicted_roles) + f"  {'-'*8}")

        for tr in true_roles:
            row = f"  {tr:<20}"
            for pr in predicted_roles:
                c = matrix[tr].get(pr, 0)
                t = totals[tr]
                pct = c / max(1, t) * 100
                row += f"  {pct:>5.1f}% ({c:>3})"
            row += f"  {totals[tr]:>8}"
            print(row)

        # Write matrix CSV for this model
        csv_path = out / f"role_confusion_matrix_{name.replace(' ', '_').replace('/', '_')}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true_role \\ predicted_role"] + predicted_roles + ["total"])
            for tr in true_roles:
                row = [tr]
                for pr in predicted_roles:
                    c = matrix[tr].get(pr, 0)
                    t = totals[tr]
                    pct = c / max(1, t) * 100
                    row.append(f"{pct:.1f}%")
                row.append(totals[tr])
                w.writerow(row)

    # Also write a combined CSV with all models
    with open(out / "role_confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "true_role"] + predicted_roles + ["total"])

        for name, _, filename in top_models:
            path = results_dir / filename
            with open(path, encoding="utf-8") as fi:
                data = json.load(fi)

            matrix = defaultdict(lambda: defaultdict(int))
            totals = defaultdict(int)

            for r in data["results"]:
                if r.get("error"):
                    continue
                qtype = r["question_type"]
                if qtype not in role_qtypes:
                    continue
                true_role = role_qtypes[qtype]
                totals[true_role] += 1
                if r["is_correct"]:
                    matrix[true_role][true_role] += 1
                    continue
                selected_idx = r.get("model_selected_index")
                if selected_idx is None or selected_idx < 0:
                    matrix[true_role]["unknown/cross-video"] += 1
                    continue
                answers = r.get("answers", [])
                if selected_idx >= len(answers):
                    matrix[true_role]["unknown/cross-video"] += 1
                    continue
                ann = ann_map.get(r["video_name"])
                if not ann:
                    matrix[true_role]["unknown/cross-video"] += 1
                    continue
                predicted = classify_selection(answers[selected_idx], ann)
                matrix[true_role][predicted] += 1

            for tr in true_roles:
                row = [name, tr]
                for pr in predicted_roles:
                    c = matrix[tr].get(pr, 0)
                    t = totals[tr]
                    pct = c / max(1, t) * 100
                    row.append(f"{pct:.1f}%")
                row.append(totals[tr])
                w.writerow(row)

    print(f"\nWrote: {out / 'role_confusion_matrix.csv'}")
    print(f"Wrote: per-model CSVs in {out}/")


if __name__ == "__main__":
    main()
