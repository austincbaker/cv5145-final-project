#!/usr/bin/env python3
"""
Analyze correlation between number of people in a scene and model accuracy.

Groups videos by person count (aggressor/victim/bystander) and computes
accuracy within each bucket for the top N models.

Usage:
    python other_scripts/person_count_correlation.py \
        --annotations annotations.json \
        --results-dir combined_results \
        --top-n 5
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def count_people(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        valid = [v for v in value if v and str(v).strip() and str(v).strip().lower() not in ('none', 'n/a', 'unknown', '')]
        return len(valid)
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s or s.lower() in ('none', 'n/a', 'unknown', ''):
        return 0
    return 1


def bucket(n: int) -> str:
    if n == 0:
        return "0"
    elif n == 1:
        return "1"
    elif n == 2:
        return "2"
    else:
        return "3+"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--results-dir", default="combined_results")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as f:
        anns = json.load(f)

    ann_map = {}
    for a in anns:
        vname = a.get("file_name") or a.get("video_name")
        ann_map[vname] = {
            "num_aggressors": count_people(a.get("aggressor")),
            "num_victims": count_people(a.get("victim")),
            "num_bystanders": count_people(a.get("bystanders")),
        }
        ann_map[vname]["total_people"] = sum(ann_map[vname].values())

    SECONDARY = {"compound_action_location", "role_count_aggressor", "role_count_victim",
                  "role_count_bystander", "compound_aggressor_victim_count", "compound_victim_bystander_count"}

    top_models = [
        ("InternVL2.5-78B-AWQ", "InternVL2.5-78B-AWQ_combined.json"),
        ("Qwen3-VL-8B-Instruct", "qwen3_8B_combined.json"),
        ("InternVL2.5-8B", "InternVL2.5-8B_combined.json"),
        ("Ovis2.5-9B", "Ovis2.5-9B_combined.json"),
        ("InternVL3.5-8B-DoT", "internvl3_5_dot_combined.json"),
    ][:args.top_n]

    roles = [
        ("Aggressors", "num_aggressors"),
        ("Victims", "num_victims"),
        ("Bystanders", "num_bystanders"),
        ("Total People", "total_people"),
    ]

    for role_label, role_key in roles:
        print(f"\n{'='*80}")
        print(f"  Accuracy by {role_label} Count (primary questions only)")
        print(f"{'='*80}")

        buckets_seen = set()
        model_data = {}

        for model_name, fname in top_models:
            path = Path(args.results_dir) / fname
            with open(path, encoding="utf-8") as f:
                d = json.load(f)

            results = d.get("results", [])
            primary = [r for r in results if r.get("question_type") not in SECONDARY]

            by_bucket = defaultdict(lambda: {"total": 0, "correct": 0})
            for r in primary:
                vname = r.get("video_name")
                info = ann_map.get(vname)
                if not info:
                    continue
                b = bucket(info[role_key])
                buckets_seen.add(b)
                by_bucket[b]["total"] += 1
                if r.get("is_correct"):
                    by_bucket[b]["correct"] += 1

            model_data[model_name] = by_bucket

        bucket_order = ["0", "1", "2", "3+"]
        buckets_used = [b for b in bucket_order if b in buckets_seen]

        header = f"{'Model':<25s}"
        for b in buckets_used:
            header += f" | {b:>8s}"
        header += f" | {'Delta':>8s}"
        print(header)
        print("-" * len(header))

        for model_name, _ in top_models:
            by_bucket = model_data[model_name]
            row = f"{model_name:<25s}"
            accs = []
            for b in buckets_used:
                d = by_bucket.get(b, {"total": 0, "correct": 0})
                if d["total"] > 0:
                    acc = d["correct"] / d["total"] * 100
                    accs.append(acc)
                    row += f" | {acc:>5.1f}% ({d['total']:>4d})"
                else:
                    row += f" |    N/A     "
            if len(accs) >= 2:
                delta = accs[-1] - accs[0]
                row += f" | {delta:>+5.1f}pp"
            else:
                row += f" |      "
            print(row)

        # Question count row
        row = f"{'(question count)':<25s}"
        all_totals = defaultdict(int)
        for model_name, _ in top_models[:1]:
            by_bucket = model_data[model_name]
            for b in buckets_used:
                all_totals[b] = by_bucket.get(b, {}).get("total", 0)
        for b in buckets_used:
            row += f" | {all_totals[b]:>10d}"
        print(row)


if __name__ == "__main__":
    main()
