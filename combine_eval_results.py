#!/usr/bin/env python3
"""
Combine per-part evaluation result JSONs into a single summary.

Use this after running `generate_questions_local.py --split N`, evaluating
each part separately (text_only_eval.sbatch or all_model_multi_gpu.sbatch),
and saving each part's `evaluation_*.json` output. This script reads each,
concatenates the `results` list, re-aggregates the per-type / per-trick
counts from scratch so the merged numbers are exact (not averages), and
writes a combined summary JSON.

Supports both output formats:
  * text_only_eval.py  -> flat summary with `total_questions`,
                          `accuracy_by_type`, `accuracy_by_trick`.
  * parallel_runner    -> `primary_*` + `secondary_*` split with
                          `video_stats` and `results`.

Usage:
    python combine_eval_results.py part1.json part2.json part3.json -o combined.json
    python combine_eval_results.py results_*/evaluation_*.json -o combined.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Mirror parallel_runner.merge_checkpoints: primary vs secondary is
# classified by question_type membership in SECONDARY_QUESTION_TYPES, not
# by an is_secondary field on the result dict (that field is never written
# into per-result records). Import the set directly so this combiner stays
# in lock-step with the generator's definition.
sys.path.insert(0, str(Path(__file__).parent))
try:
    from prompt_generator.templates import SECONDARY_QUESTION_TYPES
except ImportError:
    # Fallback if the package isn't importable (shouldn't happen in normal
    # repo layout). Leave empty -> everything counts as primary.
    SECONDARY_QUESTION_TYPES: frozenset[str] = frozenset()


def _is_parallel_runner_format(data: dict) -> bool:
    return "primary_total_questions" in data or "primary_accuracy_by_type" in data


def _combine_text_only(summaries: list[dict]) -> dict:
    """Merge text_only_eval.py summaries. Re-aggregates from the raw `results`
    lists so type / trick splits are exact, not weighted averages.
    """
    all_results: list[dict] = []
    model_paths: set[str] = set()
    for s in summaries:
        all_results.extend(s.get("results", []))
        mp = s.get("model_path")
        if mp:
            model_paths.add(mp)

    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct"))
    letter_parsed = sum(
        1 for r in all_results if r.get("model_selected_index") is not None
    )

    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_trick: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in all_results:
        qtype = r.get("question_type", "unknown")
        by_type[qtype]["total"] += 1
        if r.get("is_correct"):
            by_type[qtype]["correct"] += 1
        trick_key = "trick" if r.get("is_trick") else "normal"
        by_trick[trick_key]["total"] += 1
        if r.get("is_correct"):
            by_trick[trick_key]["correct"] += 1

    accuracy_by_type = {
        qtype: {
            "total": c["total"],
            "correct": c["correct"],
            "accuracy": c["correct"] / c["total"] if c["total"] > 0 else 0.0,
        }
        for qtype, c in by_type.items()
    }
    accuracy_by_trick = {
        key: {
            "total": c["total"],
            "correct": c["correct"],
            "accuracy": c["correct"] / c["total"] if c["total"] > 0 else 0.0,
        }
        for key, c in by_trick.items()
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "model_path": sorted(model_paths)[0] if len(model_paths) == 1 else sorted(model_paths),
        "mode": "text_only",
        "num_frames": 0,
        "num_parts_combined": len(summaries),
        "total_questions": total,
        "correct_count": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "letter_parsed_rate": letter_parsed / total if total > 0 else 0.0,
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_trick": accuracy_by_trick,
        "results": all_results,
    }


def _combine_parallel_runner(summaries: list[dict]) -> dict:
    """Merge parallel_runner summaries. Rebuilds primary/secondary stats by
    iterating the concatenated `results` list so the split is exact."""
    all_results: list[dict] = []
    model_paths: set[str] = set()
    num_frames_set: set[int] = set()
    video_stats_merged: dict[str, dict] = {}
    for s in summaries:
        all_results.extend(s.get("results", []))
        mp = s.get("model_path")
        if mp:
            model_paths.add(mp)
        nf = s.get("num_frames")
        if nf is not None:
            num_frames_set.add(nf)
        # video_stats is a per-video dict; merge across parts (no duplicates
        # expected because videos are partitioned exclusively).
        for vid, v in (s.get("video_stats") or {}).items():
            video_stats_merged[vid] = v

    def _partition(results: list[dict]) -> tuple[list[dict], list[dict]]:
        # Match parallel_runner.merge_checkpoints: classify by question_type,
        # not by a per-record is_secondary field (which isn't persisted in
        # the result dicts the evaluator emits).
        primary = [
            r for r in results
            if r.get("question_type") not in SECONDARY_QUESTION_TYPES
        ]
        secondary = [
            r for r in results
            if r.get("question_type") in SECONDARY_QUESTION_TYPES
        ]
        return primary, secondary

    def _type_stats(results: list[dict]) -> dict:
        by_type = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in results:
            qtype = r.get("question_type", "unknown")
            by_type[qtype]["total"] += 1
            if r.get("is_correct", False):
                by_type[qtype]["correct"] += 1
        return {
            qtype: {
                "total": c["total"], "correct": c["correct"],
                "accuracy": c["correct"] / c["total"] if c["total"] > 0 else 0.0,
            }
            for qtype, c in by_type.items()
        }

    primary, secondary = _partition(all_results)
    p_total = len(primary)
    p_correct = sum(1 for r in primary if r.get("is_correct"))
    s_total = len(secondary)
    s_correct = sum(1 for r in secondary if r.get("is_correct"))

    return {
        "timestamp": datetime.now().isoformat(),
        "model_path": sorted(model_paths)[0] if len(model_paths) == 1 else sorted(model_paths),
        "num_frames": next(iter(num_frames_set)) if num_frames_set else None,
        "num_parts_combined": len(summaries),
        "primary_total_questions": p_total,
        "primary_correct_count": p_correct,
        "primary_accuracy": p_correct / p_total if p_total > 0 else 0.0,
        "primary_accuracy_by_type": _type_stats(primary),
        "secondary_total_questions": s_total,
        "secondary_correct_count": s_correct,
        "secondary_accuracy": s_correct / s_total if s_total > 0 else 0.0,
        "secondary_accuracy_by_type": _type_stats(secondary),
        "total_videos_evaluated": len(video_stats_merged),
        "video_stats": video_stats_merged,
        "results": all_results,
    }


def _preprocess_raw_results(summaries: list[dict]) -> list[dict]:
    """Preprocess results that have model_response (1-based number) but no
    is_correct field.  Computes correctness from model_response vs
    correct_index and converts to the text_only format."""
    import re
    for s in summaries:
        for r in s.get("results", []):
            if "is_correct" in r:
                continue
            resp = str(r.get("model_response", "")).strip()
            match = re.search(r"\b(\d+)\b", resp)
            if match:
                sel = int(match.group(1)) - 1
                answers = r.get("answers", [])
                if 0 <= sel < len(answers):
                    r["model_selected_index"] = sel
                    r["is_correct"] = (sel == r.get("correct_index", -1))
                else:
                    r["model_selected_index"] = None
                    r["is_correct"] = False
            else:
                r["model_selected_index"] = None
                r["is_correct"] = False
            if "is_trick" not in r:
                r["is_trick"] = False
    return summaries


def combine(paths: list[Path], output_path: Path, raw: bool = False) -> dict:
    summaries: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    if raw:
        summaries = _preprocess_raw_results(summaries)

    formats = {"parallel" if _is_parallel_runner_format(s) else "text_only" for s in summaries}
    if len(formats) > 1:
        raise ValueError(
            f"Mixed result formats across inputs: {formats}. All inputs must "
            f"be produced by the same eval script."
        )
    fmt = next(iter(formats))

    if fmt == "parallel":
        merged = _combine_parallel_runner(summaries)
    else:
        merged = _combine_text_only(summaries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"Combined {len(summaries)} result file(s) [{fmt} format]:")
    for p in paths:
        print(f"  {p}")
    print(f"Wrote: {output_path}")
    if fmt == "text_only":
        print(f"  total_questions: {merged['total_questions']}")
        print(f"  accuracy:        {merged['accuracy'] * 100:.2f}%")
        print(f"  letter parsed:   {merged['letter_parsed_rate'] * 100:.1f}%")
        print(f"  by_question_type:")
        for qt, s in sorted(merged["accuracy_by_type"].items()):
            print(f"    {qt:40s}: {s['accuracy'] * 100:5.1f}%  ({s['correct']:4d}/{s['total']:4d})")
        print(f"  by_trick:")
        for k, s in sorted(merged["accuracy_by_trick"].items()):
            print(f"    {k:10s}: {s['accuracy'] * 100:5.1f}%  ({s['correct']:4d}/{s['total']:4d})")
    else:
        print(f"  primary_total_questions:   {merged['primary_total_questions']}")
        print(f"  primary_accuracy:          {merged['primary_accuracy'] * 100:.2f}%")
        print(f"  secondary_total_questions: {merged['secondary_total_questions']}")
        print(f"  secondary_accuracy:        {merged['secondary_accuracy'] * 100:.2f}%")

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more evaluation_*.json result files to merge.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Combined output JSON path. Defaults to 'combined_<timestamp>.json' "
             "in the current directory.",
    )
    parser.add_argument(
        "-r", "--raw",
        action="store_true",
        help="Input files use raw format (model_response is a 1-based number "
             "string, no is_correct field). Computes correctness automatically.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            parser.error(f"Input not found: {p}")

    if args.output:
        out = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(f"combined_{ts}.json")

    combine(paths, out, raw=args.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
