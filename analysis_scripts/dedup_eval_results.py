#!/usr/bin/env python3
"""Deduplicate eval results from parallel part files with shared checkpoints.

When eval parts run in parallel with a shared checkpoint path, later parts
accumulate questions from earlier parts. This script deduplicates by
(video_name, prompt) within each stage and recomputes all aggregates.

Works with both detailed files (with per_question) and can run on the cluster
or locally.

Usage:
    python analysis_scripts/dedup_eval_results.py train_model/experiments/20pct_adaptive_v1/eval/
    python analysis_scripts/dedup_eval_results.py train_model/experiments/5pct_adaptive_v1/eval/ --dry-run
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_parts(eval_dir: Path) -> tuple[dict[str, list[dict]], int]:
    """Load all part files and return per-stage per_question lists.

    Tries detailed files first, falls back to summary files.
    Returns (stage_data, n_parts).
    """
    detailed = sorted(eval_dir.glob("results_part*_detailed.json"))
    summary = sorted(eval_dir.glob("results_part*of*.json"))
    summary = [s for s in summary if "_detailed" not in s.name]

    files = detailed if detailed else summary
    if not files:
        raise FileNotFoundError(f"No part files found in {eval_dir}")

    stage_pqs: dict[str, list[dict]] = defaultdict(list)
    n_parts = len(files)

    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        container = data.get("stages", data)
        for stage_name, stage_data in container.items():
            if not isinstance(stage_data, dict):
                continue
            pq = stage_data.get("per_question", [])
            if pq:
                stage_pqs[stage_name].extend(pq)

    return dict(stage_pqs), n_parts


def deduplicate(per_question: list[dict]) -> list[dict]:
    """Deduplicate by (video_name, prompt), keeping the first occurrence."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for pq in per_question:
        key = (pq["video_name"], pq["prompt"])
        if key not in seen:
            seen.add(key)
            deduped.append(pq)
    return deduped


def recompute_aggregates(per_question: list[dict]) -> dict:
    """Recompute by_question_type, by_trick, by_video from per_question."""
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_trick = defaultdict(lambda: {"correct": 0, "total": 0})
    by_video = defaultdict(lambda: {"correct": 0, "total": 0})

    correct = 0
    total = 0
    letter_parsed = 0

    for pq in per_question:
        total += 1
        is_correct = pq.get("is_correct", False)
        if is_correct:
            correct += 1
        if pq.get("model_selected_index") is not None:
            letter_parsed += 1

        qt = pq.get("question_type", "unknown")
        by_type[qt]["total"] += 1
        by_type[qt]["correct"] += int(is_correct)

        trick_key = "trick" if pq.get("is_trick", False) else "normal"
        by_trick[trick_key]["total"] += 1
        by_trick[trick_key]["correct"] += int(is_correct)

        by_video[pq["video_name"]]["total"] += 1
        by_video[pq["video_name"]]["correct"] += int(is_correct)

    acc = correct / max(1, total) * 100
    lp_rate = letter_parsed / max(1, total) * 100

    return {
        "overall_accuracy": acc,
        "letter_parsed_rate": lp_rate,
        "by_question_type": dict(by_type),
        "by_trick": dict(by_trick),
        "by_video": dict(by_video),
        "total_samples": total,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("eval_dir", help="Directory with part result files")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path for merged results (default: eval_dir/results_merged.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    stage_pqs, n_parts = load_parts(eval_dir)

    if not stage_pqs:
        print("No per_question data found. Need detailed result files.")
        return

    print(f"Loaded {n_parts} part files from {eval_dir}")
    print()

    merged = {}
    for stage_name in ["base", "sft", "cot", "dpo", "adpo"]:
        if stage_name not in stage_pqs:
            continue

        raw = stage_pqs[stage_name]
        deduped = deduplicate(raw)
        removed = len(raw) - len(deduped)
        agg = recompute_aggregates(deduped)

        print(
            f"  {stage_name:>5s}: {len(raw):,} raw -> {len(deduped):,} unique "
            f"({removed:,} duplicates removed) -- "
            f"accuracy: {agg['overall_accuracy']:.2f}%"
        )

        merged[stage_name] = agg

    output_path = Path(args.output) if args.output else eval_dir / "results_merged.json"

    if not args.dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"\nWrote: {output_path}")
    else:
        print("\n(dry run -- no file written)")


if __name__ == "__main__":
    main()
