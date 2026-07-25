#!/usr/bin/env python3
"""
Analyze which distractor types the top models choose when they answer incorrectly.

For each model, when the model picks a wrong answer, classify that answer's
hardness level using the option_hardness field from the question bank. This
shows whether models are fooled by role reversals, wrong actions, bystander
substitutions, etc.

Usage:
    python analysis_scripts/distractor_type_analysis.py \
        --questions train_model/data/generated_questions.json \
        --results-dir combined_results/ \
        --top-n 5
"""

import argparse
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path


MODEL_FILE_MAP = {
    "InternVL2.5-78B-AWQ": "InternVL2.5-78B-AWQ_combined.json",
    "InternVL2.5-8B": "InternVL2.5-8B_combined.json",
    "InternVL3-9B": "InternVL3-9B_combined.json",
    "InternVL3.5-8B": "internvl3_5_combined.json",
    "InternVL3.5-8B-CoT": "internvl3_5_cot_combined.json",
    "InternVL3.5-8B-DoT": "internvl3_5_dot_combined.json",
    "InternVideo2.5-8B": "InternVideo2_5_Chat_8B_combined.json",
    "LLaVA-Video-7B": "LLaVA-Video-7B-Qwen2_combined.json",
    "Ovis2.5-9B": "Ovis2.5-9B_combined.json",
    "Ovis2.5-9B-Thinking": "Ovis2.5-9B-Thinking_combined.json",
    "Qwen2.5-VL-7B": "qwen2_5_7B_combined.json",
    "Qwen2.5-VL-72B": "qwen2_5_72B_combined.json",
    "Qwen3-VL-8B": "qwen3_8B_combined.json",
    "Qwen3-VL-8B-Thinking": "qwen3_8B_thinking_combined.json",
    "VideoLLaMA3-7B": "VideoLLaMA3-7B_combined.json",
    "gemma-4-26B": "gemma_combined.json",
}


def build_hardness_lookup(questions_path: str) -> dict:
    """Build lookup: (video_name, question_type, correct_answer) -> list of option_hardness."""
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    lookup = {}
    for video_name, questions in data["questions_by_video"].items():
        for q in questions:
            if q.get("is_secondary"):
                continue
            key = (video_name, q["question_type"], q.get("correct_index", -1))
            lookup[key] = {
                "option_hardness": q.get("option_hardness", []),
                "answers": q.get("answers", []),
            }
    return lookup


def analyze_model(results_path: str, hardness_lookup: dict) -> dict:
    """For a single model, count which distractor types it selects when wrong."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    distractor_counts = Counter()
    distractor_by_qtype = defaultdict(Counter)
    total_wrong = 0
    total_correct = 0
    unmatched = 0

    for r in data["results"]:
        if r.get("error") or r.get("model_selected_index") is None:
            continue

        if r["is_correct"]:
            total_correct += 1
            continue

        total_wrong += 1
        selected_idx = r["model_selected_index"]
        key = (r["video_name"], r["question_type"], r.get("correct_index", -1))
        q_info = hardness_lookup.get(key)

        if not q_info or not q_info["option_hardness"]:
            unmatched += 1
            continue

        hardness_list = q_info["option_hardness"]
        if selected_idx < len(hardness_list):
            h = hardness_list[selected_idx]
            distractor_counts[h] += 1
            distractor_by_qtype[r["question_type"]][h] += 1

    return {
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "unmatched": unmatched,
        "distractor_counts": distractor_counts,
        "distractor_by_qtype": dict(distractor_by_qtype),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="train_model/data/generated_questions.json")
    parser.add_argument("--results-dir", default="combined_results/")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top models to analyze")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    hardness_lookup = build_hardness_lookup(args.questions)
    print(f"Loaded hardness data for {len(hardness_lookup)} questions")

    results_dir = Path(args.results_dir)

    overall_accuracies = []
    for name, filename in MODEL_FILE_MAP.items():
        path = results_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        correct = sum(1 for r in data["results"] if r.get("is_correct"))
        total = sum(1 for r in data["results"] if not r.get("error"))
        overall_accuracies.append((name, correct / max(1, total) * 100, filename))

    overall_accuracies.sort(key=lambda x: -x[1])
    top_models = overall_accuracies[:args.top_n]

    print(f"\nTop {args.top_n} models:")
    for name, acc, _ in top_models:
        print(f"  {acc:5.1f}%  {name}")

    hardness_order = [
        "role_reversal", "wrong_action", "wrong_victim", "wrong_aggressor",
        "bystander_substitution", "wrong_location", "wrong_category",
        "none_claim", "other_in_cast", "cross_video", "frequency_saturation",
    ]

    all_results = {}
    for name, acc, filename in top_models:
        path = results_dir / filename
        result = analyze_model(str(path), hardness_lookup)
        all_results[name] = result

    # Print distractor type table
    print(f"\n{'Distractor Type':<25}", end="")
    for name, _, _ in top_models:
        short = name[:12]
        print(f"  {short:>12}", end="")
    print()
    print("-" * (25 + 14 * len(top_models)))

    for h in hardness_order:
        print(f"{h:<25}", end="")
        for name, _, _ in top_models:
            counts = all_results[name]["distractor_counts"]
            total_wrong = all_results[name]["total_wrong"]
            c = counts.get(h, 0)
            pct = c / max(1, total_wrong) * 100
            print(f"  {pct:>5.1f}% ({c:>4})", end="")
        print()

    print("-" * (25 + 14 * len(top_models)))
    print(f"{'Total wrong':<25}", end="")
    for name, _, _ in top_models:
        tw = all_results[name]["total_wrong"]
        print(f"  {tw:>11}", end="")
    print()

    # Per question-type breakdown for each model
    for name, _, _ in top_models:
        result = all_results[name]
        print(f"\n{'='*60}")
        print(f"{name} -- distractor types by question type")
        print(f"{'='*60}")
        for qtype in sorted(result["distractor_by_qtype"].keys()):
            counts = result["distractor_by_qtype"][qtype]
            total = sum(counts.values())
            print(f"\n  {qtype} ({total} wrong):")
            for h in hardness_order:
                c = counts.get(h, 0)
                if c > 0:
                    pct = c / total * 100
                    print(f"    {h:<25}: {pct:5.1f}% ({c})")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["distractor_type"] + [name for name, _, _ in top_models]
            writer.writerow(header)
            for h in hardness_order:
                row = [h]
                for name, _, _ in top_models:
                    c = all_results[name]["distractor_counts"].get(h, 0)
                    row.append(c)
                writer.writerow(row)
        print(f"\nWrote CSV: {args.output}")


if __name__ == "__main__":
    main()
