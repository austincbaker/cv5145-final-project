#!/usr/bin/env python3
"""
Extract questions from a generated-questions file that do not appear in
another (baseline) file. Produces a new questions file that the eval
pipeline can run directly, so only the delta needs to be evaluated.

Typical workflow:
    1. append_compound_action_aggressor.py -f questions.json -o questions_v2.json
    2. extract_new_questions.py -f questions_v2.json --baseline questions.json -o new_only.json
    3. Run the model eval on new_only.json (produces its own checkpoint + results)
    4. Merge results afterward

Usage:
    python extract_new_questions.py -f questions_v2.json --baseline questions.json
    python extract_new_questions.py -f questions_v2.json --baseline questions.json -o delta.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_questions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def question_key(q: dict) -> tuple:
    return (q["video_name"], q["question_type"])


def main():
    parser = argparse.ArgumentParser(
        description="Extract questions present in a file but absent from a baseline"
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Path to the full questions JSON (e.g. the _v2 file)",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline questions JSON (the original file before new questions)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path (default: <input_stem>_new_only.json)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        p = Path(args.file)
        output_path = str(p.parent / f"{p.stem}_new_only{p.suffix}")

    print(f"Loading full file: {args.file}")
    full = load_questions(args.file)
    full_qbv = full.get("questions_by_video", {})

    print(f"Loading baseline:  {args.baseline}")
    baseline = load_questions(args.baseline)
    baseline_qbv = baseline.get("questions_by_video", {})

    baseline_keys: set[tuple] = set()
    for video, qs in baseline_qbv.items():
        for q in qs:
            baseline_keys.add(question_key(q))

    delta_qbv: dict[str, list[dict]] = {}
    type_counts: defaultdict[str, int] = defaultdict(int)
    total_new = 0

    for video, qs in full_qbv.items():
        new_qs = [q for q in qs if question_key(q) not in baseline_keys]
        if new_qs:
            delta_qbv[video] = new_qs
            total_new += len(new_qs)
            for q in new_qs:
                type_counts[q["question_type"]] += 1

    if total_new == 0:
        print("No new questions found. The files have the same question set.")
        sys.exit(0)

    output = {
        "metadata": {
            "version": full.get("metadata", {}).get("version", "2.0"),
            "generation_method": "delta_extraction",
            "source_file": args.file,
            "baseline_file": args.baseline,
            "num_videos": len(delta_qbv),
            "num_questions": total_new,
            "num_distractors": full.get("metadata", {}).get("num_distractors", 7),
            "trick_probability": full.get("metadata", {}).get("trick_probability", 0.0),
            "hardness_profile": full.get("metadata", {}).get("hardness_profile", "balanced"),
            "question_counts_by_type": dict(type_counts),
        },
        "questions_by_video": delta_qbv,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted {total_new} new questions across {len(delta_qbv)} videos")
    print("By type:")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
