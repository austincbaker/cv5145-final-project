#!/usr/bin/env python3
"""
Enrich video annotations with demographic and environmental metadata.

For each video, shows the model 8 frames + existing annotation and asks
structured questions about age, gender, setting, and time of day. This is
data gathering (not evaluation) — the model's answers augment the annotations.

Runs all 2687 videos through a single model per invocation. Results from
multiple models can be merged later for consensus.

Usage:
    python analysis_scripts/enrich_annotations.py \
        --model OpenGVLab/InternVL2_5-8B \
        --annotations annotations.json \
        -o analysis_scripts/output/enriched_InternVL2.5-8B.json

    # Or on the cluster via sbatch:
    bash analysis_scripts/sbatch/enrich_all.sh
"""
import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image


ENRICHMENT_PROMPT = """\
You are analyzing a video to gather additional metadata about the people and setting.

Here is what we already know from human annotations:
- Aggressor: {aggressor}
- Victim: {victim}
- Bystanders: {bystanders}
- Action: {action}
- Environment: {environment}

Based on the video frames AND the annotations above, answer the following questions. Respond ONLY with a JSON object in this exact format:

{{
  "aggressor_age": "<adult or child>",
  "aggressor_gender": "<male or female or unknown>",
  "victim_age": "<adult or child>",
  "victim_gender": "<male or female or unknown>",
  "bystander_age": "<adult or child or mixed or none>",
  "bystander_gender": "<male or female or mixed or unknown or none>",
  "setting": "<specific setting description, or indoors/outdoors if unclear>",
  "time_of_day": "<day or night or unknown>"
}}

Rules:
- For age, choose "adult" or "child" based on apparent age. For groups, generalize to the majority.
- For gender, use "male" or "female" if identifiable. Use "unknown" if you cannot determine. For groups, use "mixed" if both are present.
- If there is no aggressor, victim, or bystander, use "none" for their age and gender fields.
- For setting, be as specific as possible (e.g., "school hallway", "parking lot", "restaurant"). Fall back to "indoors" or "outdoors" only if you cannot determine more detail.
- For time_of_day, use "day" or "night" based on lighting and context. Use "unknown" if you cannot determine."""


def load_cached_frames(video_name: str, frames_dir: str, num_frames: int = 8) -> list:
    stem = Path(video_name).stem
    frame_dir = Path(frames_dir) / stem
    if not frame_dir.exists():
        return []
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))[:num_frames]
    return [Image.open(f).convert("RGB") for f in frame_files]


def parse_json_response(response: str) -> dict | None:
    response = response.strip()
    match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--frames-dir", default="train_model/data/frames")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=25)
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

    needs_location = 0
    for i, ann in enumerate(annotations[start_idx:], start=start_idx):
        video_name = ann.get("file_name", ann.get("video_name"))
        env = ann.get("environment") or ""
        if isinstance(env, list):
            env = ", ".join(str(e) for e in env if e)
        has_location = bool(env.strip()) and env.strip().lower() not in ("", "none", "n/a")

        aggressor = ann.get("aggressor") or "none"
        victim = ann.get("victim") or "none"
        bystanders = ann.get("bystanders") or "none"
        action = ann.get("action") or "none"

        if isinstance(aggressor, list):
            aggressor = ", ".join(str(a) for a in aggressor if a)
        if isinstance(victim, list):
            victim = ", ".join(str(v) for v in victim if v)
        if isinstance(bystanders, list):
            bystanders = ", ".join(str(b) for b in bystanders if b)

        prompt = ENRICHMENT_PROMPT.format(
            aggressor=aggressor or "none",
            victim=victim or "none",
            bystanders=bystanders or "none",
            action=action or "none",
            environment=env if has_location else "not annotated",
        )

        frames = load_cached_frames(video_name, args.frames_dir, args.num_frames)
        if not frames:
            print(f"  [{i+1}/{len(annotations)}] SKIP {video_name} (no frames)")
            results.append({
                "video_name": video_name,
                "enrichment": None,
                "error": "no frames available",
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
            print(f"  [{i+1}/{len(annotations)}] ERROR {video_name}: {e}")
            results.append({
                "video_name": video_name,
                "enrichment": None,
                "error": str(e),
            })
            continue

        elapsed = time.time() - t0
        parsed = parse_json_response(response)

        if parsed:
            if not has_location:
                needs_location += 1
            preview = f"agg={parsed.get('aggressor_age','?')}/{parsed.get('aggressor_gender','?')} setting={parsed.get('setting','?')[:20]}"
            print(f"  [{i+1}/{len(annotations)}] {video_name} ({elapsed:.1f}s): {preview.encode('ascii', 'replace').decode()}")
        else:
            preview = (response[:60] + "...") if response else "EMPTY"
            print(f"  [{i+1}/{len(annotations)}] {video_name} ({elapsed:.1f}s): PARSE FAIL - {preview.encode('ascii', 'replace').decode()}")

        results.append({
            "video_name": video_name,
            "enrichment": parsed,
            "raw_response": response if not parsed else None,
            "had_location": has_location,
            "elapsed_seconds": round(elapsed, 1),
        })

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    successful = sum(1 for r in results if r.get("enrichment"))
    failed = sum(1 for r in results if not r.get("enrichment"))

    output = {
        "metadata": {
            "model": args.model,
            "num_videos": len(results),
            "num_successful": successful,
            "num_failed": failed,
            "num_frames": args.num_frames,
        },
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nDone. {successful}/{len(results)} successful.")
    print(f"Wrote: {args.output}")

    if successful > 0:
        from collections import Counter
        ages = Counter()
        genders = Counter()
        settings = Counter()
        times = Counter()
        for r in results:
            e = r.get("enrichment")
            if not e:
                continue
            ages[e.get("aggressor_age", "?")] += 1
            genders[e.get("aggressor_gender", "?")] += 1
            settings[e.get("setting", "?")[:30]] += 1
            times[e.get("time_of_day", "?")] += 1

        print(f"\nAggressor age: {dict(ages)}")
        print(f"Aggressor gender: {dict(genders)}")
        print(f"Time of day: {dict(times)}")
        print(f"Top 10 settings: {dict(settings.most_common(10))}")


if __name__ == "__main__":
    main()
