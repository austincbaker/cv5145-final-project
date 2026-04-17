#!/usr/bin/env python3
"""
Phase 1: Evaluate SFT Baseline.

Measure accuracy on test/val set:
  - Overall accuracy
  - Per-question-type breakdown
  - Trick question performance
"""

import json
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Force unbuffered output so SLURM .out files update in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _find_adapter(model_path: str) -> str | None:
    """Find adapter_config.json in model_path or its best checkpoint subdir."""
    p = Path(model_path)
    if (p / "adapter_config.json").exists():
        return str(p)
    checkpoints = sorted(p.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1]))
    for ckpt in reversed(checkpoints):
        if (ckpt / "adapter_config.json").exists():
            return str(ckpt)
    return None


def evaluate_sft(
    model_path: str,
    test_data: str,
):
    """Evaluate SFT model on test set using the language backbone."""
    base_model_name = "OpenGVLab/InternVL2_5-8B"

    print(f"[eval] Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = full_model.language_model
    print(f"[eval] Base model loaded, language backbone: {type(model).__name__}")

    print(f"[eval] Looking for LoRA adapter in {model_path}")
    adapter_path = _find_adapter(model_path)
    if adapter_path:
        print(f"[eval] Found adapter at {adapter_path}, loading...")
        model = PeftModel.from_pretrained(model, adapter_path, device_map="auto")
        print(f"[eval] LoRA weights loaded successfully")
    else:
        print(f"[eval] WARNING: No adapter found — evaluating base model only")

    with open(test_data) as f:
        examples = json.load(f)

    print(f"[eval] Starting inference on {len(examples)} examples")

    results = {
        "total": 0,
        "correct": 0,
        "by_question_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_is_trick": {"trick": {"total": 0, "correct": 0}, "non_trick": {"total": 0, "correct": 0}},
        "examples": [],
    }

    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for i, ex in enumerate(examples):
            prompt = f"{ex['video_context']}\n\nQuestion: {ex['prompt']}\n\nAnswer:"
            correct_answer = ex["correct_answer"]

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            is_correct = (
                response.lower() == correct_answer.lower()
                or correct_answer.lower() in response.lower()
            )

            qtype = ex["question_type"]
            is_trick = ex.get("is_trick", False)

            results["total"] += 1
            if is_correct:
                results["correct"] += 1

            results["by_question_type"][qtype]["total"] += 1
            if is_correct:
                results["by_question_type"][qtype]["correct"] += 1

            trick_key = "trick" if is_trick else "non_trick"
            results["by_is_trick"][trick_key]["total"] += 1
            if is_correct:
                results["by_is_trick"][trick_key]["correct"] += 1

            results["examples"].append({
                "video_name": ex["video_name"],
                "question_type": qtype,
                "is_trick": is_trick,
                "correct_answer": correct_answer,
                "predicted_answer": response,
                "is_correct": is_correct,
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(examples):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(examples) - i - 1) / rate if rate > 0 else 0
                acc = results["correct"] / results["total"] * 100
                print(
                    f"[eval] {i+1:4d}/{len(examples)} "
                    f"| acc: {acc:5.1f}% "
                    f"| {rate:.1f} ex/s "
                    f"| ETA: {eta/60:.1f}min"
                )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    overall_acc = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    print(f"\nOverall Accuracy: {results['correct']}/{results['total']} ({overall_acc:.1f}%)")

    print("\nBy Question Type:")
    for qtype in sorted(results["by_question_type"].keys()):
        stats = results["by_question_type"][qtype]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {qtype:40s}: {stats['correct']:4d}/{stats['total']:4d} ({acc:5.1f}%)")

    print("\nBy Question Category:")
    for category in ["non_trick", "trick"]:
        stats = results["by_is_trick"][category]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        cat_name = "Regular" if category == "non_trick" else "Trick"
        print(f"  {cat_name:40s}: {stats['correct']:4d}/{stats['total']:4d} ({acc:5.1f}%)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="plan2_models/sft_baseline")
    parser.add_argument("--test-data", default="plan2_data/sft_test.json")
    parser.add_argument("--output", default="plan2_eval/sft_baseline_results.json")
    args = parser.parse_args()

    results = evaluate_sft(args.model, args.test_data)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        results_to_save = {k: v for k, v in results.items() if k != "examples"}
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")
