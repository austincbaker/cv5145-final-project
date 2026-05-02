#!/usr/bin/env python3
"""
B2: Per-action-type accuracy for top N models.

Shows whether rare actions (headbutt, tackle) have worse accuracy than
common ones (punch, shove, kick).

Usage:
    python analysis_scripts/action_accuracy.py -o analysis_scripts/output/csv/
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--results-dir", default="combined_results/")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("-o", "--output-dir", default="analysis_scripts/output/csv")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, encoding="utf-8") as f:
        anns = json.load(f)
    ann_map = {a["file_name"]: a for a in anns}

    results_dir = Path(args.results_dir)

    # Find top N models by overall accuracy
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

    # Per-action accuracy for each model
    all_model_data = {}
    for name, _, filename in top_models:
        path = results_dir / filename
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        by_action = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in data["results"]:
            if r.get("error"):
                continue
            video = r["video_name"]
            ann = ann_map.get(video)
            if not ann:
                continue
            action = (ann.get("action") or "none").strip().lower()
            by_action[action]["total"] += 1
            if r["is_correct"]:
                by_action[action]["correct"] += 1

        all_model_data[name] = dict(by_action)

    # Get action counts for sorting
    action_counts = defaultdict(int)
    for a in anns:
        act = (a.get("action") or "none").strip().lower()
        action_counts[act] += 1

    all_actions = sorted(action_counts.keys(), key=lambda a: -action_counts[a])

    # Print and write CSV
    print(f"\n{'Action':<35} {'Clips':>5}", end="")
    for name, _, _ in top_models:
        print(f"  {name[:12]:>12}", end="")
    print()
    print("-" * (42 + 14 * len(top_models)))

    with open(out / "action_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["action", "clip_count"] + [name for name, _, _ in top_models]
        w.writerow(header)

        for action in all_actions:
            row = [action, action_counts[action]]
            print(f"{action:<35} {action_counts[action]:>5}", end="")
            for name, _, _ in top_models:
                data = all_model_data[name].get(action, {"correct": 0, "total": 0})
                if data["total"] > 0:
                    acc = data["correct"] / data["total"] * 100
                    print(f"  {acc:>11.1f}%", end="")
                    row.append(f"{acc:.1f}")
                else:
                    print(f"  {'--':>12}", end="")
                    row.append("")
            print()
            w.writerow(row)

    print(f"\nWrote: {out / 'action_accuracy.csv'}")


if __name__ == "__main__":
    main()
