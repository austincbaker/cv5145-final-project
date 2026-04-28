#!/usr/bin/env python3
"""
Run counting (MCQ) + social appropriateness (free-form) evaluation in one pass.

Loads the model once, then for each video:
  1. Answers counting MCQ questions (letter-based, scored automatically)
  2. Answers the social appropriateness prompt (free-form, graded later by LLM)

Usage:
    python freeform_eval/run_counting_social.py \
        --eval-file freeform_eval/counting_social_eval.json \
        --model OpenGVLab/InternVL2_5-8B \
        -o freeform_eval/results_InternVL2.5-8B.json
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


def load_cached_frames(video_name: str, frames_dir: str, num_frames: int = 8) -> list:
    stem = Path(video_name).stem
    frame_dir = Path(frames_dir) / stem
    if not frame_dir.exists():
        return []
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))[:num_frames]
    return [Image.open(f).convert("RGB") for f in frame_files]


def extract_frames(video_path: str, num_frames: int = 8) -> list:
    import subprocess, tempfile, os
    tmpdir = tempfile.mkdtemp()
    cmd = ["ffmpeg", "-i", video_path, "-frames:v", str(num_frames * 3), "-q:v", "2", f"{tmpdir}/frame_%03d.jpg", "-y", "-loglevel", "error"]
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


def format_mcq_prompt(question: dict) -> str:
    lines = [question["prompt"], "", "Options:"]
    for i, a in enumerate(question["answers"]):
        lines.append(f"  {chr(65+i)}) {a}")
    lines.append("")
    lines.append("Answer with ONLY the letter (A, B, C, or D).")
    return "\n".join(lines)


def parse_letter(response: str, num_options: int) -> int | None:
    valid = [chr(65 + i) for i in range(num_options)]
    match = re.search(r"\b([A-D])\b", response.upper())
    if match and match.group(1) in valid:
        return ord(match.group(1)) - 65
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--frames-dir", default="train_model/data/frames")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    with open(args.eval_file, encoding="utf-8") as f:
        data = json.load(f)

    questions_by_video = data["questions_by_video"]
    social_prompts = data["social_prompts"]
    video_names = sorted(set(list(questions_by_video.keys()) + list(social_prompts.keys())))

    from prompt_generator.evaluation.model_loader.registry import get_loader_class
    from prompt_generator.evaluation.model_loader.base import ModelConfig

    config = ModelConfig(model_path=args.model, num_frames=args.num_frames, max_new_tokens=256, do_sample=False)
    loader_cls = get_loader_class(args.model)
    loader = loader_cls(config)
    print(f"Loading model: {args.model}")
    loader.load()
    print(f"Model loaded. Memory: {loader.get_memory_usage()['allocated_mb']:.0f}MB")

    mcq_results = []
    social_results = []
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        mcq_results = ckpt.get("mcq_results", [])
        social_results = ckpt.get("social_results", [])
        processed = set(r["video_name"] for r in social_results)
        video_names = [v for v in video_names if v not in processed]
        print(f"Resuming: {len(processed)} videos already done, {len(video_names)} remaining")

    for i, vname in enumerate(video_names):
        frames = load_cached_frames(vname, args.frames_dir, args.num_frames)
        if not frames:
            vpath = str(Path(args.videos_dir) / vname)
            if Path(vpath).exists():
                frames = extract_frames(vpath, args.num_frames)
        if not frames:
            print(f"  [{i+1}/{len(video_names)}] SKIP {vname} (no frames)")
            continue

        # MCQ counting questions
        if vname in questions_by_video:
            for q in questions_by_video[vname]:
                prompt = format_mcq_prompt(q)
                try:
                    response = loader.generate_response(images=frames, prompt=prompt, max_new_tokens=16)
                    selected = parse_letter(response, len(q["answers"]))
                    is_correct = selected == q["correct_index"] if selected is not None else False
                    mcq_results.append({
                        "video_name": vname,
                        "question_type": q["question_type"],
                        "correct_index": q["correct_index"],
                        "model_selected_index": selected,
                        "model_response": response.strip(),
                        "is_correct": is_correct,
                    })
                except Exception as e:
                    mcq_results.append({"video_name": vname, "question_type": q["question_type"], "error": str(e), "is_correct": False})

        # Social appropriateness free-form
        if vname in social_prompts:
            sp = social_prompts[vname]
            try:
                t0 = time.time()
                response = loader.generate_response(images=frames, prompt=sp["prompt"], max_new_tokens=256)
                elapsed = time.time() - t0
                social_results.append({
                    "video_name": vname,
                    "prompt": sp["prompt"],
                    "response": response,
                    "elapsed_seconds": round(elapsed, 1),
                    "ground_truth": sp["ground_truth"],
                })
                print(f"  [{i+1}/{len(video_names)}] {vname}: {response[:70]}...")
            except Exception as e:
                social_results.append({"video_name": vname, "response": None, "error": str(e), "ground_truth": sp["ground_truth"]})
                print(f"  [{i+1}/{len(video_names)}] ERROR {vname}: {e}")

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"mcq_results": mcq_results, "social_results": social_results}, f)

    # Aggregate MCQ results
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in mcq_results:
        qt = r.get("question_type", "unknown")
        by_type[qt]["total"] += 1
        if r.get("is_correct"):
            by_type[qt]["correct"] += 1
    mcq_accuracy = {qt: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0} for qt, v in by_type.items()}

    total_mcq = sum(v["total"] for v in by_type.values())
    total_correct = sum(v["correct"] for v in by_type.values())

    output = {
        "metadata": {
            "model": args.model,
            "num_frames": args.num_frames,
            "videos_processed": len(set(r["video_name"] for r in social_results)),
        },
        "counting_mcq": {
            "total_questions": total_mcq,
            "total_correct": total_correct,
            "accuracy": total_correct / total_mcq if total_mcq else 0,
            "accuracy_by_type": mcq_accuracy,
        },
        "social_freeform": {
            "total_prompts": len(social_results),
            "successful": sum(1 for r in social_results if r.get("response")),
        },
        "mcq_results": mcq_results,
        "social_results": social_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Counting MCQ: {total_correct}/{total_mcq} ({total_correct/total_mcq*100:.1f}%)")
    for qt, v in sorted(mcq_accuracy.items()):
        print(f"  {qt:35s}: {v['accuracy']*100:5.1f}% ({v['correct']}/{v['total']})")
    print(f"Social free-form: {output['social_freeform']['successful']}/{output['social_freeform']['total_prompts']} responses collected")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
