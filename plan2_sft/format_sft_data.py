#!/usr/bin/env python3
"""
Phase 1: Format questions into SFT training data.

Converts the benchmark questions into input-output pairs suitable for SFT:
  Input: video_context + question_prompt
  Output: correct_answer (Phase 1; CoT added in Phase 2 for compounds)

Splits into train/val/test (80/10/10) stratified by question type.
"""

import json
from pathlib import Path
from collections import defaultdict
import random


def _to_text(value) -> str:
    """Normalize a field that may be str, list, int, or None to a clean string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " and ".join(str(v).strip() for v in value if v)
    return str(value).strip()


def build_video_context(annotation: dict) -> str:
    """Build a text description of the video from annotations."""
    lines = []

    aggressor = _to_text(annotation.get("aggressor"))
    victim = _to_text(annotation.get("victim"))
    action = _to_text(annotation.get("action"))
    environment = _to_text(annotation.get("environment"))
    bystanders = _to_text(annotation.get("bystanders"))

    if aggressor:
        lines.append(f"Aggressor: {aggressor}")
    if victim:
        lines.append(f"Victim: {victim}")
    if action and action.lower() != "none":
        lines.append(f"Action: {action}")
    if environment:
        lines.append(f"Environment: {environment}")
    if bystanders and "group of people" not in bystanders.lower():
        lines.append(f"Bystanders: {bystanders}")

    return "\n".join(lines) if lines else "Video context not available"


def format_sft_data(
    questions_path: str = "plan2_data/generated_questions_plan2.json",
    annotations_path: str = "annotations.json",
    output_dir: str = "plan2_data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Format questions into SFT training data with train/val/test split."""
    random.seed(seed)

    # Load questions
    with open(questions_path) as f:
        questions_data = json.load(f)
    questions_by_video = questions_data["questions_by_video"]

    # Load annotations (keyed by video_name)
    with open(annotations_path) as f:
        annotations_list = json.load(f)
    annotations_map = {a.get("file_name", a.get("video_name")): a for a in annotations_list}

    print(f"Loaded {len(questions_by_video)} videos with questions")
    print(f"Loaded {len(annotations_map)} annotations")

    # Build SFT examples: (video_name, question_type, question_index) → full example
    examples_by_qtype = defaultdict(list)
    sft_examples = []

    for video_name, questions in questions_by_video.items():
        annotation = annotations_map.get(video_name)
        if not annotation:
            # Try alternate keys
            annotation = next((a for a in annotations_list if a.get("file_name") == video_name or a.get("video_name") == video_name), None)

        if not annotation:
            print(f"Warning: no annotation for {video_name}, skipping")
            continue

        video_context = build_video_context(annotation)

        for q_idx, q in enumerate(questions):
            if q.get("is_secondary"):
                continue

            qtype = q["question_type"]

            # SFT example format for Phase 1 (direct answers, no CoT yet)
            example = {
                "video_name": video_name,
                "question_type": qtype,
                "question_index": q_idx,
                "is_trick": q.get("is_trick", False),
                "video_context": video_context,
                "prompt": q["prompt"],
                "correct_answer": q["correct_answer"],
                "all_answers": q["answers"],
                "correct_index": q.get("correct_index", -1),
            }
            sft_examples.append(example)
            examples_by_qtype[qtype].append(example)

    print(f"Total SFT examples: {len(sft_examples)}")
    print("Examples per question type:")
    for qtype, exs in sorted(examples_by_qtype.items(), key=lambda x: -len(x[1])):
        print(f"  {qtype}: {len(exs)}")

    # Stratified split by question type
    train_examples = []
    val_examples = []
    test_examples = []

    for qtype, exs in examples_by_qtype.items():
        random.shuffle(exs)
        n = len(exs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_examples.extend(exs[:n_train])
        val_examples.extend(exs[n_train:n_train + n_val])
        test_examples.extend(exs[n_train + n_val:])

    print(f"\nSplit:")
    print(f"  Train: {len(train_examples)} ({len(train_examples)/len(sft_examples)*100:.1f}%)")
    print(f"  Val:   {len(val_examples)} ({len(val_examples)/len(sft_examples)*100:.1f}%)")
    print(f"  Test:  {len(test_examples)} ({len(test_examples)/len(sft_examples)*100:.1f}%)")

    # Save splits
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }

    for split_name, examples in splits.items():
        output_file = Path(output_dir) / f"sft_{split_name}.json"
        with open(output_file, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Wrote {output_file}")

    # Summary
    summary = {
        "total_examples": len(sft_examples),
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "test_count": len(test_examples),
        "examples_by_qtype": {qtype: len(exs) for qtype, exs in examples_by_qtype.items()},
    }

    return summary


if __name__ == "__main__":
    format_sft_data()
