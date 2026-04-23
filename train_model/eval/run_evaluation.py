#!/usr/bin/env python3
"""Phase 6 — multimodal evaluation across the three pipeline stages.

Uses InternVLChatModel.chat(tokenizer, pixel_values, question, ...) so the
model actually sees frames during generation. Each adapter is loaded onto the
full InternVLChatModel (not the language_model submodule), matching how the
training scripts save adapters.

Usage:
    python train_model/eval/run_evaluation.py --config train_model/configs/eval.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from train_model.common.config import load_config
from train_model.common.video_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    register_image_context_token,
)


# claude_mcq_proposal.md Gap C: robust letter parser. The ad-hoc substring
# matcher missed "The answer is B.", "(B)", "Answer: B", and bare "B". The
# regex matches the first standalone A–H (case-insensitive) optionally
# followed by `)`, `.`, or `:`. If parsing fails, scoring falls back to
# substring text match.
LETTER_RE = re.compile(r"\b([A-H])\b\s*[\)\.\:]?", re.IGNORECASE)


def parse_letter(resp: str) -> int | None:
    """Return 0..7 for an A..H letter in `resp`, or None if no letter found."""
    m = LETTER_RE.search(resp.strip())
    return (ord(m.group(1).upper()) - ord("A")) if m else None


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[s]


def _find_adapter(path: str | Path) -> str | None:
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return str(p)
    cks = sorted(p.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1]))
    for c in reversed(cks):
        if (c / "adapter_config.json").exists():
            return str(c)
    return None


def _load_frames(frames_dir: Path, video_name: str, n: int,
                 transform: transforms.Compose, dtype: torch.dtype) -> torch.Tensor:
    stem = Path(video_name).stem
    d = frames_dir / stem
    imgs = []
    for i in range(n):
        with Image.open(d / f"frame_{i:02d}.jpg") as img:
            imgs.append(transform(img.convert("RGB")))
    return torch.stack(imgs).to(dtype=dtype)


def _build_question(example: dict, n_frames: int) -> str:
    """Build the user-turn prompt: frames + question only.

    The `video_context` annotation is deliberately omitted at eval time too so
    we measure what the model actually learned from the frames, not its ability
    to echo back the annotator's text.
    """
    frame_lines = "\n".join(f"Frame {i+1}: <image>" for i in range(n_frames))
    content = f"{frame_lines}\n\nQuestion: {example['prompt']}"
    if example.get("all_answers"):
        content += "\nOptions:"
        for i, opt in enumerate(example["all_answers"]):
            letter = chr(ord('A') + i)
            content += f"\n{letter}) {opt}"
    return content


def load_model(model_name: str, adapter_path: str | None, dtype: torch.dtype,
               tokenizer):
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.img_context_token_id = register_image_context_token(tokenizer)
    model.config.use_cache = True
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  Loaded adapter: {adapter_path}", flush=True)
    else:
        print("  WARNING: no adapter; evaluating base model", flush=True)
    model.eval()
    return model


def evaluate_stage(stage_name: str, model_path: str, cfg: dict, tokenizer,
                   test_examples: list[dict]) -> dict:
    print(f"\n{'=' * 72}\nEvaluating: {stage_name} ({model_path})\n{'=' * 72}", flush=True)
    dtype = _dtype_from_str(cfg["model"]["torch_dtype"])
    n_frames = int(cfg["video"]["frames_per_video"])
    frames_dir = Path(cfg["video"]["frames_dir"])
    image_size = int(cfg["video"]["image_size"])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    adapter_path = _find_adapter(model_path) if Path(model_path).exists() else None
    model = load_model(cfg["model"]["name"], adapter_path, dtype, tokenizer)

    gen_cfg = {
        "max_new_tokens": int(cfg["generation"].get("max_new_tokens", 64)),
        "do_sample": bool(cfg["generation"].get("do_sample", False)),
    }

    correct = 0
    total = 0
    letter_parsed = 0  # Gap C diagnostic: how often did the model emit a letter at all?
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_trick = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for i, ex in enumerate(test_examples):
            try:
                pv = _load_frames(frames_dir, ex["video_name"], n_frames,
                                  transform, dtype).to(next(model.parameters()).device)
            except FileNotFoundError:
                continue

            question = _build_question(ex, n_frames)
            try:
                response = model.chat(
                    tokenizer, pv, question, generation_config=gen_cfg,
                    num_patches_list=[1] * n_frames,
                )
            except Exception as e:
                print(f"  [{i}] chat error: {e}", flush=True)
                continue

            correct_answer_text = ex["correct_answer"].lower().strip()
            resp = response.strip()
            resp_lower = resp.lower()

            correct_idx = ex.get("correct_index", -1)
            if correct_idx == -1 and ex.get("all_answers"):
                try:
                    correct_idx = [a.lower().strip() for a in ex["all_answers"]].index(correct_answer_text)
                except ValueError:
                    pass

            # Gap C: letter-first scoring. If the model emitted a letter, that
            # is the authoritative answer. If no letter, fall back to text
            # substring match so a model that ignores the MCQ format isn't
            # silently marked 0% correct.
            parsed = parse_letter(resp)
            if parsed is not None:
                letter_parsed += 1
                is_correct = (correct_idx != -1 and parsed == correct_idx)
            elif correct_idx != -1:
                is_correct = correct_answer_text in resp_lower
            else:
                is_correct = resp_lower == correct_answer_text or correct_answer_text in resp_lower

            total += 1
            if is_correct:
                correct += 1
            qt = ex.get("question_type", "unknown")
            by_type[qt]["total"] += 1
            by_type[qt]["correct"] += int(is_correct)
            trick_key = "trick" if ex.get("is_trick", False) else "normal"
            by_trick[trick_key]["total"] += 1
            by_trick[trick_key]["correct"] += int(is_correct)

            if (i + 1) % 100 == 0:
                acc = correct / max(1, total) * 100
                lp = letter_parsed / max(1, total) * 100
                print(f"  [{i+1}/{len(test_examples)}] running acc: {acc:.1f}%  letter_parsed: {lp:.1f}%", flush=True)

    acc = correct / max(1, total) * 100
    letter_parsed_rate = letter_parsed / max(1, total) * 100
    print(f"\n  Overall: {acc:.1f}% ({correct}/{total})", flush=True)
    print(f"  Letter parsed: {letter_parsed_rate:.1f}% ({letter_parsed}/{total}) "
          f"- target >=95%; <90% means the model is not emitting letters",
          flush=True)
    for qt in sorted(by_type):
        s = by_type[qt]
        a = s["correct"] / max(1, s["total"]) * 100
        print(f"    {qt:40s}: {a:5.1f}% ({s['correct']:3d}/{s['total']:3d})", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "overall_accuracy": acc,
        "letter_parsed_rate": letter_parsed_rate,
        "by_question_type": dict(by_type),
        "by_trick": dict(by_trick),
        "total_samples": total,
    }


def run_evaluation(cfg: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(cfg["data"]["test"]) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} test examples from {cfg['data']['test']}", flush=True)

    results = {}
    for stage_name, path in cfg["stages"].items():
        if not Path(path).exists():
            print(f"  SKIP {stage_name}: {path} does not exist", flush=True)
            continue
        results[stage_name] = evaluate_stage(stage_name, path, cfg, tokenizer, examples)

    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nAblation summary:", flush=True)
    print(f"  {'stage':<20} {'overall':>8}  {'trick':>8}", flush=True)
    for s in results:
        overall = results[s]["overall_accuracy"]
        t = results[s]["by_trick"].get("trick", {})
        trick_acc = (t.get("correct", 0) / max(1, t.get("total", 1))) * 100
        print(f"  {s:<20} {overall:>7.1f}%  {trick_acc:>7.1f}%", flush=True)
    print(f"\nResults: {out_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_model/configs/eval.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
