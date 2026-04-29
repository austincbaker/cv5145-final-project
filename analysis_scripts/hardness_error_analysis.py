#!/usr/bin/env python3
"""
Analyze which distractor hardness categories fool models most often.

For each model, when the model answers incorrectly, look up the hardness
label of the option it selected. This shows which distractor types are
most confusing — role reversals, wrong actions, bystander substitutions, etc.

Usage:
    python other_scripts/hardness_error_analysis.py \
        --questions generated_questions_freq_inv_part*of3.json \
        --results-dir combined_results \
        --top-n 5
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


SECONDARY = {
    "compound_action_location", "role_count_aggressor", "role_count_victim",
    "role_count_bystander", "compound_aggressor_victim_count",
    "compound_victim_bystander_count",
}

HARDNESS_ORDER = [
    "role_reversal",
    "wrong_action",
    "wrong_victim",
    "wrong_aggressor",
    "bystander_substitution",
    "wrong_location",
    "wrong_category",
    "none_claim",
    "other_in_cast",
    "cross_video",
    "frequency_saturation",
]


def build_question_lookup(question_files: list[str]) -> dict:
    """Build lookup: (video_name, prompt) -> option_hardness list."""
    lookup = {}
    for qf in question_files:
        with open(qf, encoding="utf-8") as f:
            d = json.load(f)
        for vname, qs in d["questions_by_video"].items():
            for q in qs:
                if "option_hardness" not in q:
                    continue
                key = (q["video_name"], q["prompt"])
                lookup[key] = {
                    "option_hardness": q["option_hardness"],
                    "correct_index": q["correct_index"],
                    "answers": q["answers"],
                }
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", nargs="+", required=True)
    parser.add_argument("--results-dir", default="combined_results")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    lookup = build_question_lookup(args.questions)
    print(f"Loaded hardness labels for {len(lookup)} questions")

    top_models = [
        ("InternVL2.5-78B-AWQ", "InternVL2.5-78B-AWQ_combined.json"),
        ("Qwen3-VL-8B", "qwen3_8B_combined.json"),
        ("InternVL2.5-8B", "InternVL2.5-8B_combined.json"),
        ("Ovis2.5-9B", "Ovis2.5-9B_combined.json"),
        ("InternVL3.5-8B-DoT", "internvl3_5_dot_combined.json"),
    ][:args.top_n]

    # Part 1: When the model is WRONG, which hardness did it select?
    print(f"\n{'='*90}")
    print(f"  When models answer incorrectly, which distractor type did they fall for?")
    print(f"  (% of errors attributed to each hardness category)")
    print(f"{'='*90}")

    header = f"{'Hardness Category':<25s}"
    for model_name, _ in top_models:
        header += f" | {model_name:>18s}"
    print(header)
    print("-" * len(header))

    all_model_errors = {}
    for model_name, fname in top_models:
        path = Path(args.results_dir) / fname
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        error_by_hardness = defaultdict(int)
        total_errors = 0
        matched = 0

        for r in d["results"]:
            if r.get("question_type") in SECONDARY:
                continue
            if r.get("is_correct", False):
                continue

            sel = r.get("model_selected_index")
            if sel is None:
                continue

            key = (r["video_name"], r["prompt"])
            q = lookup.get(key)
            if not q:
                continue

            matched += 1
            hardness_list = q["option_hardness"]
            if sel < len(hardness_list):
                selected_hardness = hardness_list[sel]
                error_by_hardness[selected_hardness] += 1
                total_errors += 1

        all_model_errors[model_name] = (error_by_hardness, total_errors)

    for h in HARDNESS_ORDER:
        row = f"{h:<25s}"
        for model_name, _ in top_models:
            error_by_hardness, total_errors = all_model_errors[model_name]
            count = error_by_hardness.get(h, 0)
            if total_errors > 0:
                pct = count / total_errors * 100
                row += f" | {pct:>6.1f}% ({count:>4d})"
            else:
                row += f" |              N/A"
        print(row)

    # Totals
    row = f"{'TOTAL ERRORS':<25s}"
    for model_name, _ in top_models:
        _, total_errors = all_model_errors[model_name]
        row += f" | {total_errors:>14d}"
    print(row)

    # Part 2: Per-hardness accuracy — for questions containing each hardness
    # type as a distractor, how often does the model avoid it?
    print(f"\n{'='*90}")
    print(f"  Accuracy on questions containing each distractor type")
    print(f"  (does the model get the question right when this distractor is present?)")
    print(f"{'='*90}")

    header = f"{'Distractor Present':<25s}"
    for model_name, _ in top_models:
        header += f" | {model_name:>18s}"
    print(header)
    print("-" * len(header))

    all_model_acc = {}
    for model_name, fname in top_models:
        path = Path(args.results_dir) / fname
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        acc_by_hardness = defaultdict(lambda: {"total": 0, "correct": 0})

        for r in d["results"]:
            if r.get("question_type") in SECONDARY:
                continue

            key = (r["video_name"], r["prompt"])
            q = lookup.get(key)
            if not q:
                continue

            hardness_list = q["option_hardness"]
            present = set(hardness_list) - {"correct"}
            for h in present:
                acc_by_hardness[h]["total"] += 1
                if r.get("is_correct", False):
                    acc_by_hardness[h]["correct"] += 1

        all_model_acc[model_name] = acc_by_hardness

    for h in HARDNESS_ORDER:
        row = f"{h:<25s}"
        for model_name, _ in top_models:
            acc = all_model_acc[model_name].get(h, {"total": 0, "correct": 0})
            if acc["total"] > 0:
                pct = acc["correct"] / acc["total"] * 100
                row += f" | {pct:>6.1f}% ({acc['total']:>4d})"
            else:
                row += f" |              N/A"
        print(row)


if __name__ == "__main__":
    main()
