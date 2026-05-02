#!/usr/bin/env python3
"""
A5: Outcome annotation analysis.

Usage:
    python analysis_scripts/outcome_analysis.py -o analysis_scripts/output/csv/
"""
import argparse
import json
import csv
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("-o", "--output-dir", default="analysis_scripts/output/csv")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.annotations, encoding="utf-8") as f:
        anns = json.load(f)

    total = len(anns)
    outcomes = Counter()
    has_outcome = 0
    meaningful_outcomes = Counter()

    for a in anns:
        o = a.get("outcome")
        if o and str(o).strip() and str(o).strip().lower() not in ("", "n/a"):
            val = str(o).strip().lower()
            outcomes[val] += 1
            has_outcome += 1
            if val != "none":
                meaningful_outcomes[val] += 1
        else:
            outcomes["(no outcome field)"] += 1

    print(f"Total clips: {total}")
    print(f"With outcome field: {has_outcome} ({has_outcome/total*100:.1f}%)")
    print(f"  Of which 'none': {outcomes.get('none', 0)}")
    print(f"  Meaningful outcomes: {len(meaningful_outcomes)} unique values, {sum(meaningful_outcomes.values())} clips")

    print(f"\nOutcome distribution (meaningful only):")
    with open(out / "outcome_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["outcome", "count", "percentage_of_total"])
        for val, count in outcomes.most_common():
            pct = count / total * 100
            if count >= 2 or val in ("(no outcome field)", "none"):
                print(f"  {val:55s}: {count:4d} ({pct:5.1f}%)")
            w.writerow([val, count, f"{pct:.1f}"])

    # Comments analysis
    comments = Counter()
    for a in anns:
        for key in ("comment", "comments", "note"):
            c = a.get(key)
            if c and str(c).strip():
                comments[key] += 1

    if comments:
        print(f"\nAdditional text fields:")
        for key, count in comments.items():
            print(f"  {key}: {count} clips")

    print(f"\nWrote: {out / 'outcome_distribution.csv'}")


if __name__ == "__main__":
    main()
