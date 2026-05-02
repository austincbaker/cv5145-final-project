#!/usr/bin/env python3
"""
A1: Comprehensive dataset statistics — category distribution, action long-tail,
group structure, and clip counts.

Usage:
    python analysis_scripts/dataset_statistics.py -o analysis_scripts/output/csv/
"""
import argparse
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--groups", default="dataset.json")
    parser.add_argument("-o", "--output-dir", default="analysis_scripts/output/csv")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, encoding="utf-8") as f:
        anns = json.load(f)
    with open(args.groups, encoding="utf-8") as f:
        groups = json.load(f)

    total = len(anns)
    actions = Counter()
    for a in anns:
        act = (a.get("action") or "none").strip().lower()
        actions[act] += 1

    aggressive = sum(c for act, c in actions.items() if act != "none")
    non_aggressive = actions.get("none", 0)

    print(f"Total clips: {total}")
    print(f"Aggressive: {aggressive} ({aggressive/total*100:.1f}%)")
    print(f"Non-aggressive: {non_aggressive} ({non_aggressive/total*100:.1f}%)")
    print(f"Unique actions: {len(actions) - 1} (excluding 'none')")

    # Action distribution CSV
    print(f"\nAction distribution:")
    with open(out / "action_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "count", "percentage"])
        for act, count in actions.most_common():
            pct = count / total * 100
            print(f"  {act:35s}: {count:4d} ({pct:5.1f}%)")
            w.writerow([act, count, f"{pct:.1f}"])

    # Category/group structure CSV
    print(f"\nCategory structure:")
    with open(out / "category_structure.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "num_groups", "num_clips", "avg_clips_per_group"])
        for cat in sorted(groups.keys()):
            parents = groups[cat]
            num_groups = len(parents)
            num_clips = sum(len(clips) if isinstance(clips, list) else 1 for clips in parents.values())
            avg = num_clips / max(1, num_groups)
            print(f"  {cat:35s}: {num_groups:3d} groups, {num_clips:4d} clips ({avg:.1f} clips/group)")
            w.writerow([cat, num_groups, num_clips, f"{avg:.1f}"])

    # Long-tail summary
    sorted_actions = actions.most_common()
    non_none = [(a, c) for a, c in sorted_actions if a != "none"]
    cumulative = 0
    print(f"\nLong-tail analysis (aggressive actions only):")
    with open(out / "action_longtail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action", "count", "percentage", "cumulative_percentage"])
        for act, count in non_none:
            cumulative += count
            pct = count / aggressive * 100
            cum_pct = cumulative / aggressive * 100
            print(f"  {act:35s}: {count:4d} ({pct:5.1f}%, cumulative {cum_pct:5.1f}%)")
            w.writerow([act, count, f"{pct:.1f}", f"{cum_pct:.1f}"])

    # Top-3 concentration
    top3 = sum(c for _, c in non_none[:3])
    print(f"\nTop 3 actions ({non_none[0][0]}, {non_none[1][0]}, {non_none[2][0]}) = {top3/aggressive*100:.1f}% of aggressive clips")
    tail = sum(1 for _, c in non_none if c < 15)
    print(f"Actions with <15 clips: {tail}/{len(non_none)}")

    print(f"\nWrote CSVs to {out}/")


if __name__ == "__main__":
    main()
