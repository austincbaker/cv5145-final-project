#!/usr/bin/env python3
"""
A4: Environment distribution analysis.

Usage:
    python analysis_scripts/environment_distribution.py -o analysis_scripts/output/csv/
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
    envs = Counter()
    has_env = 0
    for a in anns:
        env = a.get("environment")
        if isinstance(env, list):
            env = ", ".join(str(e) for e in env if e)
        if env and str(env).strip() and str(env).strip().lower() not in ("", "none", "n/a"):
            envs[str(env).strip().lower()] += 1
            has_env += 1
        else:
            envs["(not annotated)"] += 1

    print(f"Total clips: {total}")
    print(f"With environment annotation: {has_env} ({has_env/total*100:.1f}%)")
    print(f"Without: {total - has_env} ({(total-has_env)/total*100:.1f}%)")
    print(f"Unique environments: {len(envs) - 1}")

    print(f"\nTop 30 environments:")
    with open(out / "environment_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["environment", "count", "percentage"])
        for env, count in envs.most_common():
            pct = count / total * 100
            if count >= 5 or env == "(not annotated)":
                print(f"  {env:45s}: {count:4d} ({pct:5.1f}%)")
            w.writerow([env, count, f"{pct:.1f}"])

    print(f"\nWrote: {out / 'environment_distribution.csv'}")


if __name__ == "__main__":
    main()
