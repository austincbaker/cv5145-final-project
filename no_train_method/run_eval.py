#!/usr/bin/env python3
"""RAP evaluation -- retrieval-augmented 1-shot prompting on InternVL2.5-8B.

Loads training data as the retrieval corpus, retrieves a same-type reference
for each test question (excluding parent-group siblings), and runs inference
with the augmented prompt.  Outputs are compatible with the existing merge
and analysis scripts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from train_model.common.video_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    register_image_context_token,
)
from train_model.eval.run_evaluation import parse_letter
from no_train_method.retriever import (
    build_parent_group_map,
    build_rap_prompt,
    build_retrieval_index,
    retrieve,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAP 1-shot evaluation")
    p.add_argument("--model", default="OpenGVLab/InternVL2_5-8B")
    p.add_argument("--train-data", required=True, help="Training questions JSON")
    p.add_argument("--test-data", required=True, help="Test questions JSON")
    p.add_argument("--dataset-json", default="dataset.json",
                   help="Parent video grouping file")
    p.add_argument("--frames-dir", default="train_model/data/frames")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("-o", "--output", required=True, help="Output JSON path")
    p.add_argument("--part", type=int, default=None,
                   help="1-indexed part number for parallel SLURM jobs")
    p.add_argument("--total-parts", type=int, default=None,
                   help="Total number of parts")
    return p.parse_args()


def load_train_data(path: str) -> list[dict]:
    """Load training questions, auto-detecting dict vs list format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions_by_video" in data:
        flat: list[dict] = []
        for video_questions in data["questions_by_video"].values():
            flat.extend(video_questions)
        return flat
    raise ValueError(
        f"Unrecognized training data format in {path}: "
        f"expected a list or a dict with 'questions_by_video' key"
    )


def load_frames(
    frames_dir: Path,
    video_name: str,
    n_frames: int,
    transform: transforms.Compose,
) -> torch.Tensor:
    stem = Path(video_name).stem
    d = frames_dir / stem
    imgs = []
    for i in range(n_frames):
        with Image.open(d / f"frame_{i:02d}.jpg") as img:
            imgs.append(transform(img.convert("RGB")))
    return torch.stack(imgs).to(dtype=torch.bfloat16)


def checkpoint_path_for(output_path: str) -> Path:
    p = Path(output_path)
    return p.parent / f"{p.stem}.checkpoint.json"


def load_checkpoint(ckpt_path: Path) -> tuple[set[tuple[str, str]], list[dict]]:
    if not ckpt_path.exists():
        return set(), []
    with open(ckpt_path, encoding="utf-8") as f:
        data = json.load(f)
    evaluated = {(r["video_name"], r["prompt"]) for r in data}
    print(
        f"  Resuming from checkpoint: {len(data)} questions already evaluated",
        flush=True,
    )
    return evaluated, data


def save_checkpoint(ckpt_path: Path, per_question: list[dict]) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(per_question, f)


def resolve_output_path(base_output: str, part: int | None,
                        total_parts: int | None) -> str:
    if part is not None and total_parts is not None:
        p = Path(base_output)
        return str(p.parent / f"{p.stem}_part{part}of{total_parts}{p.suffix}")
    return base_output


def partition_examples(
    examples: list[dict], part: int, total_parts: int,
) -> list[dict]:
    n = len(examples)
    chunk_size = (n + total_parts - 1) // total_parts
    start = (part - 1) * chunk_size
    end = min(start + chunk_size, n)
    return examples[start:end]


def main() -> None:
    args = _parse_args()

    output_path = resolve_output_path(args.output, args.part, args.total_parts)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # -- Load data -----------------------------------------------------------
    print("Loading training data...", flush=True)
    train_examples = load_train_data(args.train_data)
    print(f"  {len(train_examples)} training examples", flush=True)

    print("Loading test data...", flush=True)
    with open(args.test_data, encoding="utf-8") as f:
        test_examples: list[dict] = json.load(f)
    print(f"  {len(test_examples)} test examples", flush=True)

    if args.part is not None and args.total_parts is not None:
        test_examples = partition_examples(
            test_examples, args.part, args.total_parts,
        )
        print(
            f"  Part {args.part}/{args.total_parts}: "
            f"{len(test_examples)} examples in this partition",
            flush=True,
        )

    # -- Build retrieval index -----------------------------------------------
    print("Loading dataset grouping...", flush=True)
    with open(args.dataset_json, encoding="utf-8") as f:
        dataset = json.load(f)
    video_to_group = build_parent_group_map(dataset)
    index = build_retrieval_index(train_examples)
    print(
        f"  Retrieval index: {len(index)} (type, trick) buckets",
        flush=True,
    )

    # -- Load model ----------------------------------------------------------
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.img_context_token_id = register_image_context_token(tokenizer)
    model.config.use_cache = True
    model.eval()
    print("  Model loaded", flush=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    frames_dir = Path(args.frames_dir)

    # -- Checkpoint ----------------------------------------------------------
    ckpt_path = checkpoint_path_for(output_path)
    evaluated_keys, per_question = load_checkpoint(ckpt_path)

    correct = 0
    total = 0
    letter_parsed = 0
    retrieved_count = 0
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0},
    )

    for pq in per_question:
        total += 1
        if pq["is_correct"]:
            correct += 1
        if pq.get("model_selected_index") is not None:
            letter_parsed += 1
        if pq.get("had_reference", False):
            retrieved_count += 1
        qt = pq.get("question_type", "unknown")
        by_type[qt]["total"] += 1
        by_type[qt]["correct"] += int(pq["is_correct"])

    # -- Inference loop ------------------------------------------------------
    new_since_ckpt = 0
    with torch.no_grad():
        for i, ex in enumerate(test_examples):
            key = (ex["video_name"], ex["prompt"])
            if key in evaluated_keys:
                continue

            try:
                pv = load_frames(
                    frames_dir, ex["video_name"], args.n_frames, transform,
                ).to(next(model.parameters()).device)
            except FileNotFoundError:
                print(
                    f"  [{i}] frames not found for "
                    f"{ex['video_name'].encode('ascii', 'replace').decode()}, "
                    f"skipping",
                    flush=True,
                )
                continue

            ref = retrieve(ex, index, video_to_group)
            question = build_rap_prompt(ex, ref, args.n_frames)

            try:
                response = model.chat(
                    tokenizer,
                    pv,
                    question,
                    generation_config=gen_cfg,
                    num_patches_list=[1] * args.n_frames,
                )
            except Exception as e:
                print(
                    f"  [{i}] chat error: "
                    f"{str(e).encode('ascii', 'replace').decode()}",
                    flush=True,
                )
                continue

            resp = response.strip()
            correct_idx = ex.get("correct_index", -1)
            correct_answer_text = ex.get("correct_answer", "").lower().strip()

            parsed = parse_letter(resp)
            if parsed is not None:
                letter_parsed += 1
                is_correct = correct_idx != -1 and parsed == correct_idx
            else:
                is_correct = correct_answer_text in resp.lower()

            total += 1
            if is_correct:
                correct += 1
            if ref is not None:
                retrieved_count += 1

            qt = ex.get("question_type", "unknown")
            by_type[qt]["total"] += 1
            by_type[qt]["correct"] += int(is_correct)

            detail: dict = {
                "video_name": ex["video_name"],
                "question_type": qt,
                "prompt": ex["prompt"],
                "correct_index": correct_idx,
                "model_selected_index": parsed,
                "is_correct": is_correct,
                "model_response": resp.encode("ascii", "replace").decode(),
                "had_reference": ref is not None,
                "reference_video": ref["video_name"] if ref else None,
                "is_trick": ex.get("is_trick", False),
            }

            if not is_correct and parsed is not None and ex.get("option_hardness"):
                oh = ex["option_hardness"]
                if 0 <= parsed < len(oh):
                    detail["selected_distractor_hardness"] = oh[parsed]

            per_question.append(detail)
            new_since_ckpt += 1

            if new_since_ckpt % 100 == 0:
                save_checkpoint(ckpt_path, per_question)
                acc = correct / max(1, total) * 100
                lp = letter_parsed / max(1, total) * 100
                print(
                    f"  [{total}/{len(test_examples)}] "
                    f"running acc: {acc:.1f}%  "
                    f"letter_parsed: {lp:.1f}%  "
                    f"retrieved: {retrieved_count}/{total}  "
                    f"(checkpoint saved)",
                    flush=True,
                )

    # -- Save final output ---------------------------------------------------
    if ckpt_path.exists():
        ckpt_path.unlink()

    acc = correct / max(1, total) * 100
    lp_rate = letter_parsed / max(1, total) * 100

    print(f"\n  Overall: {acc:.1f}% ({correct}/{total})", flush=True)
    print(
        f"  Letter parsed: {lp_rate:.1f}% ({letter_parsed}/{total})",
        flush=True,
    )
    print(
        f"  Retrieved reference: {retrieved_count}/{total}",
        flush=True,
    )
    for qt in sorted(by_type):
        s = by_type[qt]
        a = s["correct"] / max(1, s["total"]) * 100
        print(
            f"    {qt:40s}: {a:5.1f}% ({s['correct']:3d}/{s['total']:3d})",
            flush=True,
        )

    output = {
        "method": "retrieval_augmented_1shot",
        "model": args.model,
        "overall_accuracy": acc,
        "letter_parsed_rate": lp_rate,
        "total_samples": total,
        "retrieved_count": retrieved_count,
        "by_question_type": dict(by_type),
        "per_question": per_question,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
