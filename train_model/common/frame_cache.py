"""One-time frame extraction for the multimodal pipeline.

Reads every video_name referenced in the SFT train/val/test splits, extracts
`frames_per_video` evenly-spaced jpgs via ffmpeg, and writes them to
`{frames_dir}/{video_stem}/frame_{i:02d}.jpg`. Skips videos that are already
fully cached.

Invoke:
    python -m train_model.common.frame_cache --config train_model/configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from train_model.common.config import load_config


def _referenced_videos(data_paths: list[str]) -> set[str]:
    videos: set[str] = set()
    for p in data_paths:
        with open(p) as f:
            examples = json.load(f)
        if isinstance(examples, list):
            for ex in examples:
                v = ex.get("video_name")
                if v:
                    videos.add(v)
        elif isinstance(examples, dict) and "questions_by_video" in examples:
            videos.update(examples["questions_by_video"].keys())
    return videos


def _video_duration(video_path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0


def _extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-ss", f"{timestamp:.3f}",
         "-i", str(video_path), "-frames:v", "1", "-q:v", "2",
         str(output_path)],
        capture_output=True, text=True,
    )
    return res.returncode == 0 and output_path.exists()


def _process_video(video_name: str, videos_dir: Path, frames_dir: Path,
                   n_frames: int) -> tuple[str, str]:
    """Return (video_name, status) where status is ok|missing|skipped|partial."""
    video_path = videos_dir / video_name
    if not video_path.is_file():
        return video_name, "missing"

    stem = Path(video_name).stem
    out_dir = frames_dir / stem

    expected = [out_dir / f"frame_{i:02d}.jpg" for i in range(n_frames)]
    if all(p.exists() for p in expected):
        return video_name, "skipped"

    duration = _video_duration(video_path)
    if duration <= 0:
        return video_name, "missing"

    ok = True
    for i, out_path in enumerate(expected):
        if out_path.exists():
            continue
        # Evenly spaced, avoiding t=0 and t=duration (which can produce junk frames)
        t = duration * (i + 1) / (n_frames + 1)
        if not _extract_frame(video_path, t, out_path):
            ok = False

    if not ok:
        return video_name, "partial"
    return video_name, "ok"


def extract_all_frames(config: dict, workers: int = 8) -> dict[str, int]:
    videos_dir = Path(config["video"]["videos_dir"])
    frames_dir = Path(config["video"]["frames_dir"])
    n_frames = int(config["video"]["frames_per_video"])

    data = config["data"]
    data_paths = [p for p in [data.get("train"), data.get("val"), data.get("test")] if p]
    videos = sorted(_referenced_videos(data_paths))
    print(f"Found {len(videos)} unique videos across {data_paths}", flush=True)
    print(f"Extracting {n_frames} frames/video into {frames_dir}", flush=True)

    counts = {"ok": 0, "skipped": 0, "missing": 0, "partial": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_video, v, videos_dir, frames_dir, n_frames): v
            for v in videos
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            name, status = fut.result()
            counts[status] += 1
            if i % 100 == 0 or i == len(videos):
                print(f"  [{i}/{len(videos)}] ok={counts['ok']} skip={counts['skipped']}"
                      f" miss={counts['missing']} partial={counts['partial']}", flush=True)

    print(f"\nDone. {counts}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train_model/configs/base.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--override", action="append", default=[],
                        help="Dotted override, e.g. video.frames_per_video=16")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    counts = extract_all_frames(cfg, workers=args.workers)
    if counts["missing"] > 0 or counts["partial"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
