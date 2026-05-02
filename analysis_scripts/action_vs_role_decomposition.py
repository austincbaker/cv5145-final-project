#!/usr/bin/env python3
"""
B4: Action recognition vs role assignment decomposition.

Compares per-model accuracy on:
  - primary_action (action only — can the model identify what happened?)
  - aggressor_identification + victim_recognition (role only — can it assign who did it?)
  - compound types (action + role combined — can it compose both?)

This decomposes errors into: action recognition error vs role assignment
error vs composition error.

Usage:
    python analysis_scripts/action_vs_role_decomposition.py -o analysis_scripts/output/csv/
"""
import argparse
import json
import csv
from pathlib import Path


MODEL_FILE_MAP = {
    "InternVL2.5-78B-AWQ": "InternVL2.5-78B-AWQ_combined.json",
    "Qwen3-VL-8B": "qwen3_8B_combined.json",
    "Ovis2.5-9B": "Ovis2.5-9B_combined.json",
    "gemma-4-26B": "gemma_combined.json",
    "InternVL3-9B": "InternVL3-9B_combined.json",
}

CATEGORIES = {
    "Action Only": ["primary_action"],
    "Role Only": ["aggressor_identification", "victim_recognition"],
    "Action + Role (compound)": [
        "compound_aggressor_action_victim",
        "compound_action_victims",
        "sequence_verification",
    ],
    "Role Pairing": ["compound_aggressor_victim"],
    "Role ID (label)": ["role_identification"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="combined_results/")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("-o", "--output-dir", default="analysis_scripts/output/csv")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
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

    print(f"\n{'Category':<30}", end="")
    for name, _, _ in top_models:
        print(f"  {name[:14]:>14}", end="")
    print()
    print("-" * (30 + 16 * len(top_models)))

    with open(out / "action_vs_role_decomposition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["category", "question_types"] + [name for name, _, _ in top_models]
        w.writerow(header)

        for cat_name, qtypes in CATEGORIES.items():
            row = [cat_name, "|".join(qtypes)]
            print(f"{cat_name:<30}", end="")

            for name, _, filename in top_models:
                path = results_dir / filename
                with open(path, encoding="utf-8") as f2:
                    data = json.load(f2)

                correct = 0
                total = 0
                for r in data["results"]:
                    if r.get("error") or r["question_type"] not in qtypes:
                        continue
                    total += 1
                    if r["is_correct"]:
                        correct += 1

                if total > 0:
                    acc = correct / total * 100
                    print(f"  {acc:>13.1f}%", end="")
                    row.append(f"{acc:.1f}")
                else:
                    print(f"  {'--':>14}", end="")
                    row.append("")

            print()
            w.writerow(row)

    # Composition gap analysis
    print(f"\n{'Composition gap (Action Only - Compound):'}")
    for name, _, filename in top_models:
        path = results_dir / filename
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        action_c, action_t, compound_c, compound_t, role_c, role_t = 0, 0, 0, 0, 0, 0
        for r in data["results"]:
            if r.get("error"):
                continue
            qt = r["question_type"]
            if qt in CATEGORIES["Action Only"]:
                action_t += 1
                action_c += int(r["is_correct"])
            elif qt in CATEGORIES["Action + Role (compound)"]:
                compound_t += 1
                compound_c += int(r["is_correct"])
            elif qt in CATEGORIES["Role Only"]:
                role_t += 1
                role_c += int(r["is_correct"])

        action_acc = action_c / max(1, action_t) * 100
        role_acc = role_c / max(1, role_t) * 100
        compound_acc = compound_c / max(1, compound_t) * 100
        gap = action_acc - compound_acc
        print(f"  {name:25s}: action={action_acc:.1f}%, role={role_acc:.1f}%, compound={compound_acc:.1f}%, gap={gap:+.1f}pp")

    print(f"\nWrote: {out / 'action_vs_role_decomposition.csv'}")


if __name__ == "__main__":
    main()
