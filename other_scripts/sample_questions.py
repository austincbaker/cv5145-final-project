#!/usr/bin/env python3
"""Randomly select two questions from each primary category and display in markdown."""

import json
import argparse
import random

SECONDARY_TYPES = {
    "role_count_aggressor", "role_count_victim", "role_count_bystander",
    "compound_aggressor_victim_count", "compound_victim_bystander_count",
    "compound_action_location",
}


def main():
    parser = argparse.ArgumentParser(description="Sample questions from each primary category")
    parser.add_argument("questions", help="Path to generated questions JSON file")
    parser.add_argument("-n", "--count", type=int, default=2, help="Number of questions per category (default: 2)")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for video, qs in data.get("questions_by_video", {}).items():
        questions.extend(qs)

    by_type = {}
    for q in questions:
        qt = q["question_type"]
        if qt not in SECONDARY_TYPES:
            by_type.setdefault(qt, []).append(q)

    if args.seed is not None:
        random.seed(args.seed)

    print("# Sample Questions (Primary Categories)\n")

    for qt in sorted(by_type.keys()):
        pool = by_type[qt]
        n = min(args.count, len(pool))
        samples = random.sample(pool, n)

        print(f"## {qt.replace('_', ' ').title()}\n")

        for i, q in enumerate(samples, 1):
            print(f"### Sample {i} — `{q['video_name']}`\n")
            print(f"**Prompt:** {q['prompt']}\n")
            print("| # | Answer | |")
            print("|---|--------|-|")
            for j, a in enumerate(q["answers"]):
                marker = " ✓" if j == q["correct_index"] else ""
                print(f"| {j + 1} | {a} |{marker} |")
            print()

        print("---\n")


if __name__ == "__main__":
    main()
