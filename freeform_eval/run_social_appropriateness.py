#!/usr/bin/env python3
"""
Run social appropriateness evaluation on all videos.

Shows each model 8 frames and asks: "How would you characterize the
appropriateness of the actions depicted in the video? Defend your reasoning."

The model generates a free-form response with max_new_tokens=512 to avoid
truncation of the reasoning.

Usage:
    python freeform_eval/run_social_appropriateness.py \
        --model OpenGVLab/InternVL2_5-8B \
        --annotations annotations.json \
        -o freeform_eval/social_responses_InternVL2.5-8B.json
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image


SOCIAL_PROMPT = "How would you characterize the appropriateness of the actions depicted in the video? Defend your reasoning."


def load_cached_frames(video_name: str, frames_dir: str, num_frames: int = 8) -> list:
    stem = Path(video_name).stem
    frame_dir = Path(frames_dir) / stem
    if not frame_dir.exists():
        return []
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))[:num_frames]
    return [Image.open(f).convert("RGB") for f in frame_files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--frames-dir", default="train_model/data/frames")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations")

    from prompt_generator.evaluation.model_loader.registry import get_loader_class
    from prompt_generator.evaluation.model_loader.base import ModelConfig

    config = ModelConfig(
        model_path=args.model,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    loader_cls = get_loader_class(args.model)
    loader = loader_cls(config)
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
        print(f"Resuming from checkpoint: {start_idx}/{len(annotations)}")

    for i, ann in enumerate(annotations[start_idx:], start=start_idx):
        video_name = ann.get("file_name", ann.get("video_name"))
        action = (ann.get("action") or "none").strip().lower()
        is_aggressive = action != "none" and action != ""

        frames = load_cached_frames(video_name, args.frames_dir, args.num_frames)
        if not frames:
            print(f"  [{i+1}/{len(annotations)}] SKIP {video_name} (no frames)")
            results.append({
                "video_name": video_name,
                "prompt": SOCIAL_PROMPT,
                "response": None,
                "error": "no frames available",
                "ground_truth": {
                    "is_aggressive": is_aggressive,
                    "action": ann.get("action"),
                    "aggressor": ann.get("aggressor"),
                    "victim": ann.get("victim"),
                    "environment": ann.get("environment"),
                    "bystanders": ann.get("bystanders"),
                },
            })
            continue

        t0 = time.time()
        try:
            response = loader.generate_response(
                images=frames,
                prompt=SOCIAL_PROMPT,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            response = None
            print(f"  [{i+1}/{len(annotations)}] ERROR {video_name}: {e}")
            results.append({
                "video_name": video_name,
                "prompt": SOCIAL_PROMPT,
                "response": None,
                "error": str(e),
                "ground_truth": {
                    "is_aggressive": is_aggressive,
                    "action": ann.get("action"),
                    "aggressor": ann.get("aggressor"),
                    "victim": ann.get("victim"),
                    "environment": ann.get("environment"),
                    "bystanders": ann.get("bystanders"),
                },
            })
            continue

        elapsed = time.time() - t0
        preview = (response[:80] + "...") if response else "EMPTY"
        print(f"  [{i+1}/{len(annotations)}] {video_name} ({elapsed:.1f}s): {preview.encode('ascii', 'replace').decode()}")

        results.append({
            "video_name": video_name,
            "prompt": SOCIAL_PROMPT,
            "response": response,
            "elapsed_seconds": round(elapsed, 1),
            "ground_truth": {
                "is_aggressive": is_aggressive,
                "action": ann.get("action"),
                "aggressor": ann.get("aggressor"),
                "victim": ann.get("victim"),
                "environment": ann.get("environment"),
                "bystanders": ann.get("bystanders"),
            },
        })

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    output = {
        "metadata": {
            "model": args.model,
            "prompt": SOCIAL_PROMPT,
            "num_videos": len(results),
            "num_successful": sum(1 for r in results if r.get("response")),
            "num_aggressive": sum(1 for r in results if r["ground_truth"]["is_aggressive"]),
            "num_non_aggressive": sum(1 for r in results if not r["ground_truth"]["is_aggressive"]),
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

    print(f"\nDone. {output['metadata']['num_successful']}/{len(results)} successful.")
    print(f"  Aggressive: {output['metadata']['num_aggressive']}")
    print(f"  Non-aggressive: {output['metadata']['num_non_aggressive']}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
