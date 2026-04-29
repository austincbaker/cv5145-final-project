#!/usr/bin/env python3
"""Generate a pipe-delimited CSV summary of combined baseline results."""
import json
import csv
import sys
from pathlib import Path

MODELS = [
    ("InternVL2.5-8B", "combined_results/InternVL2.5-8B_combined.json"),
    ("Ovis2.5-9B", "combined_results/Ovis2.5-9B_combined.json"),
    ("Ovis2.5-9B-Thinking", "combined_results/Ovis2.5-9B-Thinking_combined.json"),
    ("InternVL3-9B", "combined_results/InternVL3-9B_combined.json"),
    ("InternVideo2.5-8B", "combined_results/InternVideo2_5_Chat_8B_combined.json"),
]

OUTPUT = "combined_results/baseline_accuracy_summary.csv"


def main():
    primary_types = set()
    secondary_types = set()
    data = []

    for name, path in MODELS:
        with open(path) as f:
            d = json.load(f)
        primary_types.update(d.get("primary_accuracy_by_type", {}).keys())
        secondary_types.update(d.get("secondary_accuracy_by_type", {}).keys())
        data.append((name, d))

    primary_types = sorted(primary_types)
    secondary_types = sorted(secondary_types)

    header = ["Model", "Overall Accuracy", "Primary Accuracy"]
    header += ["Primary: " + t for t in primary_types]
    header += ["Secondary Accuracy"]
    header += ["Secondary: " + t for t in secondary_types]

    rows = []
    for name, d in data:
        p_total = d["primary_total_questions"]
        p_correct = d["primary_correct_count"]
        s_total = d["secondary_total_questions"]
        s_correct = d["secondary_correct_count"]
        total = p_total + s_total
        correct = p_correct + s_correct
        overall = correct / total * 100 if total > 0 else 0
        p_acc = d["primary_accuracy"] * 100
        s_acc = d["secondary_accuracy"] * 100

        row = [name]
        row.append("{:.2f}%".format(overall))
        row.append("{:.2f}%".format(p_acc))

        for t in primary_types:
            info = d.get("primary_accuracy_by_type", {}).get(t)
            if info:
                row.append("{:.2f}%".format(info["accuracy"] * 100))
            else:
                row.append("N/A")

        row.append("{:.2f}%".format(s_acc))

        for t in secondary_types:
            info = d.get("secondary_accuracy_by_type", {}).get(t)
            if info:
                row.append("{:.2f}%".format(info["accuracy"] * 100))
            else:
                row.append("N/A")

        rows.append(row)

    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    print("Wrote " + OUTPUT)
    print()
    print("|".join(header))
    for r in rows:
        print("|".join(r))


if __name__ == "__main__":
    main()
