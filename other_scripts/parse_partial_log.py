#!/usr/bin/env python3
"""Parse unfinished GPU evaluation logs and produce accuracy stats matching
the format of a completed evaluation JSON (summary section only)."""

import argparse
import json
import re
import sys
from collections import defaultdict

# Secondary question types (counting categories + compound_action_location)
SECONDARY_QUESTION_TYPES = {
    "role_count_aggressor",
    "role_count_victim",
    "role_count_bystander",
    "compound_aggressor_victim_count",
    "compound_victim_bystander_count",
    "compound_action_location",
}

STATUS_RE = re.compile(r"Status:\s+(CORRECT|WRONG)")
TYPE_RE = re.compile(r"Type:\s+(\S+)")
VIDEO_RE = re.compile(r"\[(\d+)/(\d+)\]\s+Processing\s+(\S+)")
GPU_RE = re.compile(r"\[GPU\s+(\d+)\]")


def parse_log(path: str) -> dict:
    primary_by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    secondary_by_type = defaultdict(lambda: {"total": 0, "correct": 0})

    videos_seen = set()
    current_type = None
    total_videos = 0

    with open(path, "r") as f:
        for line in f:
            # Track video progress
            m = VIDEO_RE.search(line)
            if m:
                idx, total, video = m.group(1), int(m.group(2)), m.group(3)
                total_videos = max(total_videos, total)
                videos_seen.add(video)
                current_type = None
                continue

            # Track question type
            m = TYPE_RE.search(line)
            if m:
                current_type = m.group(1)
                continue

            # Track result
            m = STATUS_RE.search(line)
            if m and current_type:
                is_correct = m.group(1) == "CORRECT"
                bucket = secondary_by_type if current_type in SECONDARY_QUESTION_TYPES else primary_by_type
                bucket[current_type]["total"] += 1
                if is_correct:
                    bucket[current_type]["correct"] += 1
                current_type = None

    return {
        "total_videos_assigned": total_videos,
        "videos_processed": len(videos_seen),
        "primary": primary_by_type,
        "secondary": secondary_by_type,
    }


def build_summary(data: dict) -> dict:
    primary = data["primary"]
    secondary = data["secondary"]

    p_total = sum(v["total"] for v in primary.values())
    p_correct = sum(v["correct"] for v in primary.values())

    s_total = sum(v["total"] for v in secondary.values())
    s_correct = sum(v["correct"] for v in secondary.values())

    def type_stats(d):
        out = {}
        for qtype in sorted(d):
            t, c = d[qtype]["total"], d[qtype]["correct"]
            out[qtype] = {"total": t, "correct": c, "accuracy": c / t if t else 0}
        return out

    summary = {
        "note": "PARTIAL — parsed from in-progress log",
        "videos_processed": data["videos_processed"],
        "total_videos_assigned": data["total_videos_assigned"],
        "primary_total_questions": p_total,
        "primary_correct_count": p_correct,
        "primary_accuracy": p_correct / p_total if p_total else 0,
        "primary_accuracy_by_type": type_stats(primary),
        "secondary_total_questions": s_total,
        "secondary_correct_count": s_correct,
        "secondary_accuracy": s_correct / s_total if s_total else 0,
        "secondary_accuracy_by_type": type_stats(secondary),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse partial evaluation logs into accuracy stats")
    parser.add_argument("log_files", nargs="+", help="One or more GPU stdout log files")
    parser.add_argument("-o", "--output", help="Write JSON to file instead of stdout")
    args = parser.parse_args()

    # Merge results from all GPU logs
    merged_primary = defaultdict(lambda: {"total": 0, "correct": 0})
    merged_secondary = defaultdict(lambda: {"total": 0, "correct": 0})
    all_videos = set()
    max_total = 0

    for log_file in args.log_files:
        data = parse_log(log_file)
        max_total = max(max_total, data["total_videos_assigned"])
        all_videos.update(range(data["videos_processed"]))  # just for count
        for qtype, vals in data["primary"].items():
            merged_primary[qtype]["total"] += vals["total"]
            merged_primary[qtype]["correct"] += vals["correct"]
        for qtype, vals in data["secondary"].items():
            merged_secondary[qtype]["total"] += vals["total"]
            merged_secondary[qtype]["correct"] += vals["correct"]

    # Use actual video count from all logs
    total_videos_processed = sum(parse_log(f)["videos_processed"] for f in args.log_files)

    merged = {
        "total_videos_assigned": max_total,
        "videos_processed": total_videos_processed,
        "primary": merged_primary,
        "secondary": merged_secondary,
    }

    summary = build_summary(merged)
    output = json.dumps(summary, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
