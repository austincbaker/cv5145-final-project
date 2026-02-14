#!/usr/bin/env python3
"""
Standalone question generation script — no model or GPU required.
Run locally to pre-generate the questions JSON for later evaluation.

Usage:
    python generate_questions_local.py annotations.json
    python generate_questions_local.py annotations.json -o my_questions.json
    python generate_questions_local.py annotations.json -d 7
"""
import sys
import os

# Add project root to path so we can import prompt_generator as a package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_generator.generator import QuestionGenerator
from prompt_generator.templates import QuestionType
from prompt_generator.answer_bank import normalize_entry

import argparse
import json
from pathlib import Path


# Selected question types for evaluation
SELECTED_QUESTION_TYPES = [
    # Simple
    QuestionType.AGGRESSOR_ID,
    QuestionType.VICTIM_RECOGNITION,
    # Complex
    QuestionType.COMPOUND_AGGRESSOR_VICTIM,
    QuestionType.COMPOUND_AGGRESSOR_ACTION_VICTIM,
    QuestionType.COMPOUND_ACTION_VICTIMS,
    # Summary
    QuestionType.INTERACTION_SUMMARY,
]


def load_annotations(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "annotations" in data:
            return data["annotations"]
        return [data]
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")


def generate_questions_for_all_videos(
    annotations_path: str,
    output_path: str,
    num_distractors: int = 7,
) -> dict:
    """
    Generate questions for all videos and save to JSON.
    No model loading required — purely deterministic from annotations.
    """
    annotations = load_annotations(annotations_path)
    print(f"Loaded {len(annotations)} annotations")

    generator = QuestionGenerator(annotations, num_distractors=num_distractors)

    all_questions = []
    skipped_count = 0

    total = len(annotations)
    for idx, entry in enumerate(annotations, 1):
        video_name = entry.get("video_name") or entry.get("file_name", "unknown")
        print(f"\r[{idx}/{total}] Generating questions for: {video_name}", end="", flush=True)

        for question_type in SELECTED_QUESTION_TYPES:
            question = generator.generate_question(
                entry=entry,
                question_type=question_type,
            )

            if question is not None:
                question_dict = {
                    "video_name": video_name,
                    "question_type": question.question_type,
                    "prompt": question.prompt,
                    "answers": question.answers,
                    "correct_answer": question.correct_answer,
                    "correct_index": question.correct_index,
                }
                all_questions.append(question_dict)
            else:
                skipped_count += 1

    print()  # newline after status bar

    # Group questions by video
    questions_by_video = {}
    for q in all_questions:
        video = q["video_name"]
        if video not in questions_by_video:
            questions_by_video[video] = []
        questions_by_video[video].append(q)

    # Statistics
    question_counts = {}
    for q in all_questions:
        qtype = q["question_type"]
        question_counts[qtype] = question_counts.get(qtype, 0) + 1

    output = {
        "metadata": {
            "num_videos": len(questions_by_video),
            "num_questions": len(all_questions),
            "num_distractors": num_distractors,
            "question_types": [qt.value for qt in SELECTED_QUESTION_TYPES],
            "question_counts_by_type": question_counts,
            "skipped_questions": skipped_count,
        },
        "questions_by_video": questions_by_video,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated questions:")
    print(f"  Videos: {len(questions_by_video)}")
    print(f"  Total questions: {len(all_questions)}")
    print(f"  Skipped questions: {skipped_count}")
    if questions_by_video:
        print(f"  Questions per video: {len(all_questions) / len(questions_by_video):.1f}")
    print(f"\nQuestions by type:")
    for qtype, count in sorted(question_counts.items()):
        print(f"  {qtype}: {count}")
    print(f"\nSaved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate questions for all videos (no model/GPU required)"
    )
    parser.add_argument(
        "annotations_json",
        help="Path to JSON file containing video annotations",
    )
    parser.add_argument(
        "-o", "--output",
        default="generated_questions.json",
        help="Output JSON file path (default: generated_questions.json)",
    )
    parser.add_argument(
        "-d", "--num-distractors",
        type=int,
        default=7,
        help="Number of distractor answers per question (default: 7, giving 8 total options)",
    )

    args = parser.parse_args()

    try:
        generate_questions_for_all_videos(
            args.annotations_json,
            args.output,
            args.num_distractors,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
