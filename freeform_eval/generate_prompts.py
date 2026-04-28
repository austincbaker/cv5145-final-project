#!/usr/bin/env python3
"""
Generate open-ended prompts for free-form aggression understanding evaluation.

Selects a stratified subset of videos and produces a JSON file with the
prompt and ground-truth annotations for each. The prompt asks the model
to assess social appropriateness and defend its answer — no MCQ options.

Usage:
    python freeform_eval/generate_prompts.py \
        --annotations annotations.json \
        --num-videos 100 \
        --seed 42 \
        -o freeform_eval/prompts.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


PROMPT = (
    "Based on the video, do you think the actions taking place are socially "
    "appropriate? Defend your answer by describing what you observe: who is "
    "involved, what actions are being performed, and whether any individuals "
    "appear to be aggressors or victims."
)


def select_videos(annotations: list[dict], num_videos: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    by_action = defaultdict(list)
    for a in annotations:
        action = (a.get("action") or "none").strip().lower()
        by_action[action].append(a)

    selected = []
    actions = sorted(by_action.keys())

    none_count = max(1, int(num_videos * 0.15))
    aggressive_count = num_videos - none_count

    if "none" in by_action:
        pool = by_action["none"]
        rng.shuffle(pool)
        selected.extend(pool[:none_count])

    agg_actions = [a for a in actions if a != "none"]
    per_action = max(1, aggressive_count // len(agg_actions))
    remainder = aggressive_count - per_action * len(agg_actions)

    for action in agg_actions:
        pool = by_action[action]
        rng.shuffle(pool)
        n = per_action + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        selected.extend(pool[:n])

    rng.shuffle(selected)
    return selected[:num_videos]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="freeform_eval/prompts.json")
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as f:
        annotations = json.load(f)

    selected = select_videos(annotations, args.num_videos, args.seed)

    prompts = []
    for ann in selected:
        video_name = ann.get("file_name") or ann.get("video_name")
        action = (ann.get("action") or "none").strip()
        is_aggressive = action.lower() != "none"

        ground_truth = {
            "is_aggressive": is_aggressive,
            "action": action if is_aggressive else None,
            "aggressor": ann.get("aggressor") if is_aggressive else None,
            "victim": ann.get("victim") if is_aggressive else None,
            "environment": ann.get("environment"),
            "bystanders": ann.get("bystanders"),
        }

        prompts.append({
            "video_name": video_name,
            "prompt": PROMPT,
            "ground_truth": ground_truth,
        })

    output = {
        "metadata": {
            "num_videos": len(prompts),
            "num_aggressive": sum(1 for p in prompts if p["ground_truth"]["is_aggressive"]),
            "num_non_aggressive": sum(1 for p in prompts if not p["ground_truth"]["is_aggressive"]),
            "seed": args.seed,
            "prompt_template": PROMPT,
        },
        "prompts": prompts,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(prompts)} prompts ({output['metadata']['num_aggressive']} aggressive, "
          f"{output['metadata']['num_non_aggressive']} non-aggressive)")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
