#!/usr/bin/env python3
"""
Generate compound_action_aggressor questions.

Two modes:
  --standalone  Output ONLY the new questions (one per video), for fast
                isolated eval runs.
  (default)     Append to the existing question file.

Usage:
    # Standalone file with only Action+Aggressor questions:
    python append_compound_action_aggressor.py -f generated_questions.json --standalone

    # Append to existing file:
    python append_compound_action_aggressor.py -f generated_questions.json

    # Custom output path:
    python append_compound_action_aggressor.py -f generated_questions.json --standalone -o action_aggressor_only.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_generator.answer_bank import normalize_entry
from prompt_generator.generator import QuestionGenerator
from prompt_generator.hardness import DEFAULT_RECIPES, apply_hardness_profile
from prompt_generator.templates import (
    QUESTION_CATEGORIES,
    SECONDARY_QUESTION_TYPES,
    QuestionType,
)

NEW_QTYPE = QuestionType.COMPOUND_ACTION_AGGRESSOR


def load_annotations(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "annotations" in data:
            return data["annotations"]
        return [data]
    raise ValueError(f"Unexpected JSON structure in {path}")


def default_output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_v2{p.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Append compound_action_aggressor questions to an existing questions file"
    )
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Path to the existing generated-questions JSON file",
    )
    parser.add_argument(
        "-a", "--annotations",
        default="annotations.json",
        help="Path to the annotations JSON file (default: annotations.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path (default: <input_stem>_v2.json)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Output only the new questions (not appended to existing file)",
    )
    args = parser.parse_args()

    if args.standalone and args.output is None:
        output_path = str(Path(args.file).parent / "compound_action_aggressor.json")
    else:
        output_path = args.output or default_output_path(args.file)

    # Load existing questions file
    print(f"Loading existing questions from: {args.file}")
    with open(args.file, "r", encoding="utf-8") as f:
        existing = json.load(f)

    metadata = existing.get("metadata", {})
    questions_by_video: dict[str, list] = existing.get("questions_by_video", {})

    num_distractors = metadata.get("num_distractors", 7)
    trick_probability = metadata.get("trick_probability", 0.0)
    hardness_profile = metadata.get("hardness_profile", "balanced")

    print(f"  {metadata.get('num_videos', '?')} videos, {metadata.get('num_questions', '?')} existing questions")
    print(f"  distractors={num_distractors}, trick_prob={trick_probability}, profile={hardness_profile}")

    # Load annotations and build generator
    print(f"Loading annotations from: {args.annotations}")
    annotations = load_annotations(args.annotations)
    print(f"  {len(annotations)} annotations loaded")

    recipes = apply_hardness_profile(DEFAULT_RECIPES, hardness_profile)
    generator = QuestionGenerator(
        annotations,
        num_distractors=num_distractors,
        trick_probability=trick_probability,
        recipes=recipes,
    )

    # Build video_name -> annotation lookup
    ann_by_video: dict[str, dict] = {}
    for entry in annotations:
        name = entry.get("video_name") or entry.get("file_name")
        if name:
            ann_by_video[name] = entry

    # Generate new questions
    added = 0
    skipped_no_ann = 0
    skipped_no_fields = 0
    standalone_questions: dict[str, list] = {}

    video_names = list(questions_by_video.keys())
    total = len(video_names)

    for idx, video_name in enumerate(video_names, 1):
        print(f"\r[{idx}/{total}] {video_name}", end="", flush=True)

        entry = ann_by_video.get(video_name)
        if entry is None:
            skipped_no_ann += 1
            continue

        question = generator.generate_question(entry=entry, question_type=NEW_QTYPE)
        if question is None:
            skipped_no_fields += 1
            continue

        question_dict = {
            "video_name": question.video_name,
            "question_type": question.question_type,
            "is_secondary": question.question_type in SECONDARY_QUESTION_TYPES,
            "is_trick": question.is_trick,
            "prompt": question.prompt,
            "answers": question.answers,
            "correct_answer": question.correct_answer,
            "correct_index": question.correct_index,
            "option_hardness": getattr(question, "option_hardness", None),
        }

        if args.standalone:
            standalone_questions.setdefault(video_name, []).append(question_dict)
        else:
            questions_by_video[video_name].append(question_dict)
        added += 1

    print()  # newline after progress line

    if args.standalone:
        output = {
            "metadata": {
                "num_videos": len(standalone_questions),
                "num_questions": added,
                "num_distractors": num_distractors,
                "trick_probability": trick_probability,
                "hardness_profile": hardness_profile,
                "question_counts_by_type": {NEW_QTYPE.value: added},
            },
            "questions_by_video": standalone_questions,
        }
    else:
        new_total_questions = sum(len(qs) for qs in questions_by_video.values())
        new_metadata = dict(metadata)
        new_metadata["num_questions"] = new_total_questions

        qtype_counts = dict(metadata.get("question_counts_by_type", {}))
        qtype_counts[NEW_QTYPE.value] = added
        new_metadata["question_counts_by_type"] = qtype_counts

        cat = QUESTION_CATEGORIES.get(NEW_QTYPE)
        if cat:
            cat_counts = dict(metadata.get("category_counts", {}))
            cat_counts[cat.value] = cat_counts.get(cat.value, 0) + added
            new_metadata["category_counts"] = cat_counts

        dist_config = dict(metadata.get("distribution_config", {}))
        if cat:
            dist_config[cat.value] = dist_config.get(cat.value, 0) + 1
        new_metadata["distribution_config"] = dist_config

        output = {
            "metadata": new_metadata,
            "questions_by_video": questions_by_video,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {added} compound_action_aggressor questions")
    if skipped_no_ann:
        print(f"Skipped {skipped_no_ann} videos (no matching annotation)")
    if skipped_no_fields:
        print(f"Skipped {skipped_no_fields} videos (missing action/aggressor fields)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
