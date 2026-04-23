#!/usr/bin/env python3
"""
Strip secondary-category questions from a generated_questions.json so the
text-only eval sees the same question set the normal (frame-based) eval
does.

The normal eval path uses train_model/data/sft_test.json, which is produced
by train_model/sft/format_data.py. That script filters out every question
where `is_secondary` is True (counting / compound_action_location).
text_only_eval.py reads generated_questions.json directly and therefore
evaluates those secondary questions too, inflating the denominator.

Usage:
    python make_frameless_questions.py generated_questions.json
    python make_frameless_questions.py generated_questions.json -o generated_questions_text_only.json
    python make_frameless_questions.py generated_questions.json --drop-trick

When --drop-trick is passed, trick questions (those that resolve to a
"None" / null-claim correct answer) are also removed. By default tricks
are kept to match the normal eval's behaviour.
"""
import argparse
import json
from pathlib import Path


def strip_secondary(
    questions_path: str,
    output_path: str,
    drop_trick: bool = False,
) -> dict:
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions_by_video = data.get("questions_by_video", {})
    metadata = data.get("metadata", {})

    total_before = 0
    total_after = 0
    dropped_secondary = 0
    dropped_trick = 0
    emptied_videos = 0

    filtered_by_video: dict[str, list] = {}
    for video_name, questions in questions_by_video.items():
        kept = []
        for q in questions:
            total_before += 1
            if q.get("is_secondary"):
                dropped_secondary += 1
                continue
            if drop_trick and q.get("is_trick"):
                dropped_trick += 1
                continue
            kept.append(q)
        if kept:
            filtered_by_video[video_name] = kept
            total_after += len(kept)
        else:
            emptied_videos += 1

    # Update metadata so downstream consumers see the new counts.
    new_metadata = dict(metadata)
    new_metadata["num_videos"] = len(filtered_by_video)
    new_metadata["num_questions"] = total_after
    new_metadata["frameless_filter"] = {
        "dropped_secondary": dropped_secondary,
        "dropped_trick": dropped_trick,
        "kept": total_after,
        "source": str(questions_path),
    }

    output = {
        "metadata": new_metadata,
        "questions_by_video": filtered_by_video,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Source: {questions_path}")
    print(f"  Before: {total_before} questions across {len(questions_by_video)} videos")
    print(f"  Dropped secondary: {dropped_secondary}")
    if drop_trick:
        print(f"  Dropped trick:     {dropped_trick}")
    print(f"  After:  {total_after} questions across {len(filtered_by_video)} videos")
    if emptied_videos:
        print(f"  Videos emptied by filter: {emptied_videos}")
    print(f"Saved to: {output_path}")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "questions_json",
        help="Path to the generated_questions.json to filter",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path. Defaults to <input>_text_only.json.",
    )
    parser.add_argument(
        "--drop-trick",
        action="store_true",
        help="Also drop is_trick questions (default: keep, matching normal eval)",
    )
    args = parser.parse_args()

    input_path = Path(args.questions_json)
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_name(f"{input_path.stem}_text_only.json"))

    strip_secondary(
        questions_path=str(input_path),
        output_path=output_path,
        drop_trick=args.drop_trick,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
