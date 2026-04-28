#!/usr/bin/env python3
"""
Run free-form evaluation on selected videos.

Uses the existing model loader infrastructure to generate open-ended
responses (no MCQ options). Each model sees the video frames + the prompt
and generates a free-form answer.

Usage:
    python freeform_eval/run_freeform.py \
        --prompts freeform_eval/prompts.json \
        --model OpenGVLab/InternVL2_5-8B \
        --num-frames 8 \
        --max-new-tokens 256 \
        -o freeform_eval/responses_InternVL2.5-8B.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def extract_frames(video_path: str, num_frames: int = 8) -> list:
    """Extract uniformly-spaced frames from a video file."""
    import subprocess
    import tempfile
    import os

    tmpdir = tempfile.mkdtemp()
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='not(mod(n\\,{max(1, 1)}))',setpts=N/FRAME_RATE/TB",
        "-frames:v", str(num_frames * 3),
        "-q:v", "2",
        f"{tmpdir}/frame_%03d.jpg",
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True, timeout=30)

    frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))
    if not frame_files:
        return []

    step = max(1, len(frame_files) // num_frames)
    selected = frame_files[::step][:num_frames]
    frames = [Image.open(f).convert("RGB") for f in selected]

    for f in frame_files:
        os.unlink(f)
    os.rmdir(tmpdir)
    return frames


def load_cached_frames(video_name: str, frames_dir: str, num_frames: int = 8) -> list:
    """Load pre-cached frames if available."""
    stem = Path(video_name).stem
    frame_dir = Path(frames_dir) / stem
    if not frame_dir.exists():
        return []
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))[:num_frames]
    return [Image.open(f).convert("RGB") for f in frame_files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="Path to prompts.json")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--videos-dir", default="videos", help="Directory containing video files")
    parser.add_argument("--frames-dir", default="train_model/data/frames", help="Pre-cached frames directory")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data["prompts"]

    from prompt_generator.evaluation.model_loader.registry import create_loader
    from prompt_generator.evaluation.model_loader.base import ModelConfig

    config = ModelConfig(
        model_path=args.model,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    loader = create_loader(args.model, config)
    print(f"Loading model: {args.model}")
    loader.load()
    print(f"Model loaded. Memory: {loader.get_memory_usage()['allocated_mb']:.0f}MB")

    results = []
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from checkpoint: {start_idx}/{len(prompts)}")

    for i, entry in enumerate(prompts[start_idx:], start=start_idx):
        video_name = entry["video_name"]
        prompt = entry["prompt"]

        frames = load_cached_frames(video_name, args.frames_dir, args.num_frames)
        if not frames:
            video_path = str(Path(args.videos_dir) / video_name)
            if Path(video_path).exists():
                frames = extract_frames(video_path, args.num_frames)

        if not frames:
            print(f"  [{i+1}/{len(prompts)}] SKIP {video_name} (no frames)")
            results.append({
                "video_name": video_name,
                "prompt": prompt,
                "response": None,
                "error": "no frames available",
                "ground_truth": entry["ground_truth"],
            })
            continue

        t0 = time.time()
        try:
            response = loader.generate_response(
                images=frames,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            response = None
            print(f"  [{i+1}/{len(prompts)}] ERROR {video_name}: {e}")
            results.append({
                "video_name": video_name,
                "prompt": prompt,
                "response": None,
                "error": str(e),
                "ground_truth": entry["ground_truth"],
            })
            continue

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(prompts)}] {video_name} ({elapsed:.1f}s): {response[:80]}...")

        results.append({
            "video_name": video_name,
            "prompt": prompt,
            "response": response,
            "elapsed_seconds": round(elapsed, 1),
            "ground_truth": entry["ground_truth"],
        })

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    output = {
        "metadata": {
            "model": args.model,
            "num_videos": len(results),
            "num_successful": sum(1 for r in results if r.get("response")),
            "num_frames": args.num_frames,
            "max_new_tokens": args.max_new_tokens,
        },
        "responses": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nDone. {output['metadata']['num_successful']}/{len(results)} successful responses.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
