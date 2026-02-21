#!/usr/bin/env python3
"""
Sample N questions per question type from a generated questions JSON and write
them to a markdown file for quick human review.

Usage:
    python preview_questions.py generated_questions.json
    python preview_questions.py generated_questions.json -o preview.md
    python preview_questions.py generated_questions.json -n 3 --seed 42
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from prompt_generator.templates import QUESTION_CATEGORIES


# Map each question type value → category value, and
# build an ordered list of type values per category.
TYPE_TO_CATEGORY: dict[str, str] = {}
CATEGORY_TYPE_ORDER: dict[str, list[str]] = defaultdict(list)
for _qtype, _cat in QUESTION_CATEGORIES.items():
    TYPE_TO_CATEGORY[_qtype.value] = _cat.value
    CATEGORY_TYPE_ORDER[_cat.value].append(_qtype.value)

CATEGORY_ORDER = ["simple", "compound", "complex", "counting", "identification"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for video_questions in data.get("questions_by_video", {}).values():
        questions.extend(video_questions)
    return questions


def format_question(q: dict, index: int) -> str:
    video = q.get("video_name", "unknown")
    prompt = q.get("prompt", "")
    answers = q.get("answers", [])
    correct_idx = q.get("correct_index", -1)

    lines = [
        f"**Q{index}** — `{video}`",
        f"*{prompt}*",
        "",
    ]

    for i, answer in enumerate(answers):
        letter = LETTERS[i] if i < len(LETTERS) else str(i + 1)
        if i == correct_idx:
            lines.append(f"- **{letter}. {answer}** ✓")
        else:
            lines.append(f"- {letter}. {answer}")

    lines.append("")
    return "\n".join(lines)


def build_markdown(
    questions: list[dict],
    n: int,
    seed: int | None,
    source_path: str,
) -> str:
    # Group questions by question type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q.get("question_type", "unknown")].append(q)

    rng = random.Random(seed)

    lines = [
        "# Question Preview",
        "",
        f"**Source:** `{source_path}`  ",
        f"**Sampling:** {n} per question type"
        + (f" (seed={seed})" if seed is not None else ""),
        "",
        "---",
        "",
    ]

    global_index = 1

    for category in CATEGORY_ORDER:
        type_list = CATEGORY_TYPE_ORDER.get(category, [])
        active_types = [t for t in type_list if t in by_type]
        if not active_types:
            continue

        lines.append(f"## {category.title()} Questions")
        lines.append("")

        for qtype in active_types:
            pool = by_type[qtype]
            sample = rng.sample(pool, min(n, len(pool)))

            type_label = qtype.replace("_", " ").title()
            lines.append(f"### {type_label}")
            lines.append("")

            for q in sample:
                lines.append(format_question(q, global_index))
                global_index += 1

        lines.append("---")
        lines.append("")

    lines.append(f"*{global_index - 1} questions shown.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Preview N questions per question type from a generated questions JSON"
    )
    parser.add_argument(
        "questions_json",
        help="Path to generated questions JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output markdown file (default: <input_stem>_preview.md)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=2,
        help="Number of questions to sample per question type (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )

    args = parser.parse_args()

    input_path = Path(args.questions_json)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_preview.md"
    )

    try:
        questions = load_questions(args.questions_json)
    except FileNotFoundError:
        print(f"Error: file not found: {args.questions_json}")
        return 1
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return 1

    if not questions:
        print("No questions found in the file.")
        return 1

    print(f"Loaded {len(questions)} questions")

    md = build_markdown(questions, args.n, args.seed, args.questions_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Saved preview to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
