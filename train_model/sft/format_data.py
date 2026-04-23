#!/usr/bin/env python3
"""
Phase 1: Format questions into SFT training data.

Converts the benchmark questions into input-output pairs suitable for SFT:
  Input: video_context + question_prompt
  Output: correct_answer (Phase 1; CoT added in Phase 2 for compounds)

Splits into train/val/test (80/10/10) stratified by **video** (not by
question_type), so the same video never appears in more than one split. Within
each split, question-type distribution stays roughly balanced because the
split is stratified over the video's `action` label from annotations.json.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import random


def _validate_mcq_consistency(examples: list, source: str) -> None:
    """Fail loudly if any example has MCQ fields whose correct_index and
    correct_answer disagree. Mirror of the check in
    train_model/common/video_dataset.py. Running it before splits are
    written means a broken generator run can't silently corrupt training.
    """
    for i, ex in enumerate(examples):
        answers = ex.get("all_answers")
        if not answers:
            continue
        idx = ex.get("correct_index", -1)
        if idx < 0 or idx >= len(answers):
            raise ValueError(
                f"MCQ consistency violation in {source!r} at example {i} "
                f"(video={ex.get('video_name')!r}): correct_index={idx} "
                f"out of range for {len(answers)} options"
            )
        if answers[idx] != ex["correct_answer"]:
            raise ValueError(
                f"MCQ consistency violation in {source!r} at example {i} "
                f"(video={ex.get('video_name')!r}): "
                f"all_answers[{idx}]={answers[idx]!r} != "
                f"correct_answer={ex['correct_answer']!r}"
            )


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
    questions_path: str = "train_model/data/generated_questions.json",
    annotations_path: str = "annotations.json",
    output_dir: str = "train_model/data",
    train_ratio: float = 0.2,
    val_ratio: float = 0.0,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Format questions into SFT training data with train/val/test split.

    Defaults to a 20 / 0 / 80 split (no validation). Pass --train-ratio and
    --val-ratio to adjust. test_ratio = 1 - train_ratio - val_ratio.
    When val_ratio == 0 the script still writes an empty sft_val.json so
    downstream scripts can unconditionally open it.
    """
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    output_path = Path(output_dir)
    train_file = output_path / "sft_train.json"
    val_file = output_path / "sft_val.json"
    test_file = output_path / "sft_test.json"

    if not force and train_file.exists() and val_file.exists() and test_file.exists():
        print(f"Skipping formatting: SFT data already exists in {output_dir} (use --force to regenerate)")
        with open(train_file, encoding="utf-8") as f:
            train_examples = json.load(f)
        with open(val_file, encoding="utf-8") as f:
            val_examples = json.load(f)
        with open(test_file, encoding="utf-8") as f:
            test_examples = json.load(f)
        total = len(train_examples) + len(val_examples) + len(test_examples)
        return {
            "total_examples": total,
            "train_count": len(train_examples),
            "val_count": len(val_examples),
            "test_count": len(test_examples),
            "examples_by_qtype": {},
        }

    random.seed(seed)

    # Load questions
    with open(questions_path, encoding="utf-8") as f:
        questions_data = json.load(f)
    questions_by_video = questions_data["questions_by_video"]

    # Load annotations (keyed by video_name)
    with open(annotations_path, encoding="utf-8") as f:
        annotations_list = json.load(f)
    annotations_map = {a.get("file_name", a.get("video_name")): a for a in annotations_list}

    print(f"Loaded {len(questions_by_video)} videos with questions")
    print(f"Loaded {len(annotations_map)} annotations")

    # Build SFT examples: (video_name, question_type, question_index) -> full example
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

    # Gap D safety: ensure every example's correct_index and correct_answer
    # agree before we split and write to disk. Catches generator drift early.
    _validate_mcq_consistency(sft_examples, source=questions_path)

    # -------- Video-level split (no video leakage across splits) ---------
    # Stratify the *videos* by their primary action so each split still
    # contains a similar mix of aggression types. Then fan all questions
    # from a video into the same split.

    examples_by_video = defaultdict(list)
    for ex in sft_examples:
        examples_by_video[ex["video_name"]].append(ex)

    videos_by_action = defaultdict(list)
    for video_name in examples_by_video:
        action = _to_text(annotations_map.get(video_name, {}).get("action")) or "unknown"
        videos_by_action[action.lower()].append(video_name)

    train_videos: set[str] = set()
    val_videos: set[str] = set()
    test_videos: set[str] = set()

    for action, videos in videos_by_action.items():
        random.shuffle(videos)
        n = len(videos)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_videos.update(videos[:n_train])
        val_videos.update(videos[n_train:n_train + n_val])
        test_videos.update(videos[n_train + n_val:])

    # Safety: assert disjoint splits.
    assert train_videos.isdisjoint(val_videos), "train/val video overlap"
    assert train_videos.isdisjoint(test_videos), "train/test video overlap"
    assert val_videos.isdisjoint(test_videos), "val/test video overlap"

    train_examples: list[dict] = []
    val_examples: list[dict] = []
    test_examples: list[dict] = []
    for video_name, exs in examples_by_video.items():
        if video_name in train_videos:
            train_examples.extend(exs)
        elif video_name in val_videos:
            val_examples.extend(exs)
        elif video_name in test_videos:
            test_examples.extend(exs)

    print(
        f"\nVideo-level split (stratified by action, {len(videos_by_action)} actions):"
    )
    print(
        f"  Train: {len(train_videos):4d} videos  /  {len(train_examples):5d} examples"
        f"  ({len(train_examples)/len(sft_examples)*100:.1f}%)"
    )
    print(
        f"  Val:   {len(val_videos):4d} videos  /  {len(val_examples):5d} examples"
        f"  ({len(val_examples)/len(sft_examples)*100:.1f}%)"
    )
    print(
        f"  Test:  {len(test_videos):4d} videos  /  {len(test_examples):5d} examples"
        f"  ({len(test_examples)/len(sft_examples)*100:.1f}%)"
    )

    # Save splits
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }

    for split_name, examples in splits.items():
        output_file = Path(output_dir) / f"sft_{split_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="train_model/data/generated_questions.json")
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--output-dir", default="train_model/data")
    parser.add_argument("--train-ratio", type=float, default=0.2,
                        help="Fraction of videos for training (default 0.2)")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="Fraction of videos for validation (default 0.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if output files already exist")
    args = parser.parse_args()

    format_sft_data(
        questions_path=args.questions,
        annotations_path=args.annotations,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force,
    )
