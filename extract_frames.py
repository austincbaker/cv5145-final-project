#!/usr/bin/env python3
"""Extract still frames from videos referenced by sample questions."""

import argparse
import json
import os
import random
import subprocess
import sys

SECONDARY_TYPES = {
    "role_count_aggressor", "role_count_victim", "role_count_bystander",
    "compound_aggressor_victim_count", "compound_victim_bystander_count",
    "compound_action_location",
}


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def extract_frame(video_path: str, timestamp: float, output_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
         "-frames:v", "1", "-q:v", "2", output_path],
        capture_output=True, text=True
    )


def load_action_lookup(annotations_path: str) -> dict:
    if not os.path.isfile(annotations_path):
        return {}
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {e.get("file_name"): e.get("action", "unknown") for e in data
            if e.get("file_name")}


def select_diverse_samples(pool, n, action_lookup, rng):
    """Pick n questions from pool, preferring distinct action labels."""
    if n >= len(pool):
        return list(pool)

    by_action = {}
    for q in pool:
        action = action_lookup.get(q["video_name"], "unknown")
        by_action.setdefault(action, []).append(q)

    for bucket in by_action.values():
        rng.shuffle(bucket)

    action_order = list(by_action.keys())
    rng.shuffle(action_order)

    selected = []
    while len(selected) < n and action_order:
        remaining = []
        for action in action_order:
            if len(selected) >= n:
                break
            bucket = by_action[action]
            if bucket:
                selected.append(bucket.pop())
                if bucket:
                    remaining.append(action)
        action_order = remaining

    return selected


def select_sample_questions(questions_path, annotations_path="annotations.json",
                            count_per_type=2, seed=42):
    """Return the list of sample questions used for the slides (diverse-action)."""
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    action_lookup = load_action_lookup(annotations_path)

    questions = []
    for qs in data.get("questions_by_video", {}).values():
        questions.extend(qs)

    by_type = {}
    for q in questions:
        qt = q["question_type"]
        if qt not in SECONDARY_TYPES:
            by_type.setdefault(qt, []).append(q)

    rng = random.Random(seed)
    selected = []
    for qt in sorted(by_type.keys()):
        pool = by_type[qt]
        n = min(count_per_type, len(pool))
        selected.extend(select_diverse_samples(pool, n, action_lookup, rng))
    return selected


def extract_frames_for_questions(selected_questions, videos_dir, output_dir,
                                 frames_per_video=8, verbose=True):
    """Extract `frames_per_video` still frames for each unique video.

    Skips frames that already exist. Returns (extracted_count, missing_videos).
    """
    os.makedirs(output_dir, exist_ok=True)

    videos_seen = set()
    extracted = 0
    missing = []

    for q in selected_questions:
        video_name = q["video_name"]
        if video_name in videos_seen:
            continue
        videos_seen.add(video_name)

        video_path = os.path.join(videos_dir, video_name)
        if not os.path.isfile(video_path):
            video_path = os.path.join("/home/austin/newton_code/bullying-project/", videos_dir, video_name)
            if not os.path.isfile(video_path):
                if verbose:
                    print(f"WARNING: Video not found: {video_path}", file=sys.stderr)
                missing.append(video_name)
                continue

        base_name = os.path.splitext(video_name)[0]

        needed = []
        for i in range(frames_per_video):
            output_path = os.path.join(output_dir, f"{base_name}_frame{i + 1}.jpg")
            if not os.path.isfile(output_path):
                needed.append((i, output_path))

        if not needed:
            continue

        duration = get_video_duration(video_path)
        if duration <= 0:
            if verbose:
                print(f"WARNING: Could not determine duration for {video_name}",
                      file=sys.stderr)
            continue

        for i, output_path in needed:
            timestamp = duration * (i + 1) / (frames_per_video + 1)
            extract_frame(video_path, timestamp, output_path)
            extracted += 1
            if verbose:
                print(f"Extracted: {output_path}")

    return extracted, missing


def main():
    parser = argparse.ArgumentParser(description="Extract still frames from sample question videos")
    parser.add_argument("questions", help="Path to generated questions JSON file")
    parser.add_argument("-v", "--videos-dir", default="videos", help="Path to videos directory (default: videos)")
    parser.add_argument("-o", "--output-dir", default="sample_frames", help="Output directory for frames (default: sample_frames)")
    parser.add_argument("-n", "--count", type=int, default=2, help="Number of questions per category (default: 2)")
    parser.add_argument("-f", "--frames", type=int, default=8, help="Number of frames per video (default: 8)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42, matches generate_results_powerpoint.py)")
    parser.add_argument("--annotations", default="annotations.json", help="Annotations JSON (used to diversify actions per question type)")
    args = parser.parse_args()

    selected = select_sample_questions(
        args.questions, args.annotations, args.count, args.seed
    )
    extract_frames_for_questions(
        selected, args.videos_dir, args.output_dir,
        frames_per_video=args.frames, verbose=True,
    )
    print(f"\nDone. Frames saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
