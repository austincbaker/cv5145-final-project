#!/usr/bin/env python3
"""
Temporal shuffle experiment: evaluate a model with randomly permuted frames.

If accuracy is similar to the proper-order baseline, temporal information
is not important for the model's decisions. If accuracy drops significantly,
the model relies on temporal dynamics.

Runs on a 10% random sample of the question bank by default.

Usage:
    python analysis_scripts/temporal_shuffle_eval.py \
        --model OpenGVLab/InternVL2_5-8B \
        --questions train_model/data/generated_questions.json \
        --sample-rate 0.1 \
        -o analysis_scripts/output/temporal_shuffle_InternVL2.5-8B.json
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image


def load_cached_frames(video_name: str, frames_dir: str, num_frames: int = 8) -> list:
    stem = Path(video_name).stem
    frame_dir = Path(frames_dir) / stem
    if not frame_dir.exists():
        return []
    frame_files = sorted(frame_dir.glob("frame_*.jpg"))[:num_frames]
    return [Image.open(f).convert("RGB") for f in frame_files]


def parse_letter(response: str, num_options: int) -> int | None:
    import re
    resp = response.strip()
    m = re.search(r'\b([A-H])\b\s*[\)\.\:]?', resp, re.IGNORECASE)
    if m:
        idx = ord(m.group(1).upper()) - ord('A')
        if 0 <= idx < num_options:
            return idx
    for i in range(num_options):
        if resp.startswith(str(i + 1)):
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", default="train_model/data/generated_questions.json")
    parser.add_argument("--frames-dir", default="train_model/data/frames")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.questions, encoding="utf-8") as f:
        qdata = json.load(f)

    all_questions = []
    for video_name, questions in qdata["questions_by_video"].items():
        for q in questions:
            if q.get("is_secondary"):
                continue
            q["video_name"] = video_name
            all_questions.append(q)

    sample_size = int(len(all_questions) * args.sample_rate)
    sampled = random.sample(all_questions, sample_size)
    print(f"Sampled {len(sampled)}/{len(all_questions)} questions ({args.sample_rate*100:.0f}%)")

    from prompt_generator.evaluation.model_loader.registry import get_loader_class
    from prompt_generator.evaluation.model_loader.base import ModelConfig

    config = ModelConfig(
        model_path=args.model,
        num_frames=args.num_frames,
        max_new_tokens=64,
        do_sample=False,
    )
    loader_cls = get_loader_class(args.model)
    loader = loader_cls(config)
    print(f"Loading model: {args.model}")
    loader.load()
    print(f"Model loaded. Memory: {loader.get_memory_usage()['allocated_mb']:.0f}MB")

    results_normal = []
    results_shuffled = []
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        results_normal = ckpt["normal"]
        results_shuffled = ckpt["shuffled"]
        start_idx = len(results_normal)
        print(f"Resuming from checkpoint: {start_idx}/{len(sampled)}")

    for i, q in enumerate(sampled[start_idx:], start=start_idx):
        video_name = q["video_name"]
        frames = load_cached_frames(video_name, args.frames_dir, args.num_frames)
        if not frames:
            continue

        prompt = q["prompt"]
        answers = q.get("answers", [])
        correct_idx = q.get("correct_index", -1)

        options_text = "\n".join(f"{chr(65+j)}) {a}" for j, a in enumerate(answers))
        full_prompt = f"Question: {prompt}\nOptions:\n{options_text}"

        # Normal order
        try:
            resp_normal = loader.generate_response(images=frames, prompt=full_prompt, max_new_tokens=64)
        except Exception as e:
            resp_normal = str(e)

        parsed_normal = parse_letter(resp_normal, len(answers))
        correct_normal = (parsed_normal == correct_idx) if parsed_normal is not None else False

        # Shuffled order
        shuffled_frames = frames.copy()
        random.shuffle(shuffled_frames)

        try:
            resp_shuffled = loader.generate_response(images=shuffled_frames, prompt=full_prompt, max_new_tokens=64)
        except Exception as e:
            resp_shuffled = str(e)

        parsed_shuffled = parse_letter(resp_shuffled, len(answers))
        correct_shuffled = (parsed_shuffled == correct_idx) if parsed_shuffled is not None else False

        result_entry = {
            "video_name": video_name,
            "question_type": q["question_type"],
            "correct_index": correct_idx,
        }

        results_normal.append({**result_entry, "is_correct": correct_normal, "parsed_index": parsed_normal})
        results_shuffled.append({**result_entry, "is_correct": correct_shuffled, "parsed_index": parsed_shuffled})

        if (i + 1) % 25 == 0:
            n_correct = sum(1 for r in results_normal if r["is_correct"])
            s_correct = sum(1 for r in results_shuffled if r["is_correct"])
            n_total = len(results_normal)
            print(f"  [{i+1}/{len(sampled)}] normal={n_correct}/{n_total} ({n_correct/max(1,n_total)*100:.1f}%) shuffled={s_correct}/{n_total} ({s_correct/max(1,n_total)*100:.1f}%)")

        if (i + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"normal": results_normal, "shuffled": results_shuffled}, f)

    # Summary
    n_total = len(results_normal)
    n_correct = sum(1 for r in results_normal if r["is_correct"])
    s_correct = sum(1 for r in results_shuffled if r["is_correct"])

    by_type_normal = defaultdict(lambda: {"correct": 0, "total": 0})
    by_type_shuffled = defaultdict(lambda: {"correct": 0, "total": 0})
    for rn, rs in zip(results_normal, results_shuffled):
        qt = rn["question_type"]
        by_type_normal[qt]["total"] += 1
        by_type_normal[qt]["correct"] += int(rn["is_correct"])
        by_type_shuffled[qt]["total"] += 1
        by_type_shuffled[qt]["correct"] += int(rs["is_correct"])

    print(f"\n{'='*70}")
    print(f"Temporal Shuffle Experiment -- {args.model}")
    print(f"{'='*70}")
    print(f"Sample: {n_total} questions ({args.sample_rate*100:.0f}% of benchmark)")
    print(f"\nOverall:")
    print(f"  Normal order:   {n_correct/max(1,n_total)*100:.1f}% ({n_correct}/{n_total})")
    print(f"  Shuffled order: {s_correct/max(1,n_total)*100:.1f}% ({s_correct}/{n_total})")
    print(f"  Delta:          {(s_correct-n_correct)/max(1,n_total)*100:+.1f}pp")
    print(f"\nPer question type:")
    print(f"  {'Type':<40} {'Normal':>8} {'Shuffled':>8} {'Delta':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for qt in sorted(by_type_normal.keys()):
        bn = by_type_normal[qt]
        bs = by_type_shuffled[qt]
        na = bn["correct"] / max(1, bn["total"]) * 100
        sa = bs["correct"] / max(1, bs["total"]) * 100
        print(f"  {qt:<40} {na:>7.1f}% {sa:>7.1f}% {sa-na:>+7.1f}")

    output = {
        "metadata": {
            "model": args.model,
            "sample_rate": args.sample_rate,
            "seed": args.seed,
            "num_questions": n_total,
            "num_frames": args.num_frames,
        },
        "summary": {
            "normal_accuracy": round(n_correct / max(1, n_total) * 100, 2),
            "shuffled_accuracy": round(s_correct / max(1, n_total) * 100, 2),
            "delta_pp": round((s_correct - n_correct) / max(1, n_total) * 100, 2),
            "by_question_type": {
                qt: {
                    "normal": round(by_type_normal[qt]["correct"] / max(1, by_type_normal[qt]["total"]) * 100, 2),
                    "shuffled": round(by_type_shuffled[qt]["correct"] / max(1, by_type_shuffled[qt]["total"]) * 100, 2),
                    "delta": round((by_type_shuffled[qt]["correct"] - by_type_normal[qt]["correct"]) / max(1, by_type_normal[qt]["total"]) * 100, 2),
                    "total": by_type_normal[qt]["total"],
                }
                for qt in sorted(by_type_normal.keys())
            },
        },
        "results_normal": results_normal,
        "results_shuffled": results_shuffled,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
