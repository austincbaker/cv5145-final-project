#!/usr/bin/env python3
"""
Phase 1: Evaluate SFT Baseline.

Measure accuracy on test/val set:
  - Overall accuracy
  - Per-question-type breakdown
  - Trick question performance
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


def evaluate_sft(
    model_path: str,
    test_data: str,
):
    """Evaluate SFT model on test set using the language backbone."""
    base_model_name = "OpenGVLab/InternVL2_5-8B"

    print(f"Loading model from {model_path}")
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

    if Path(model_path).exists():
        try:
            model = PeftModel.from_pretrained(model, model_path, device_map="auto")
            print(f"Loaded LoRA weights from {model_path}")
        except Exception as e:
            print(f"Warning: could not load LoRA weights: {e}")

    with open(test_data) as f:
        examples = json.load(f)

    print(f"Evaluating on {len(examples)} examples")

    results = {
        "total": 0,
        "correct": 0,
        "by_question_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_is_trick": {"trick": {"total": 0, "correct": 0}, "non_trick": {"total": 0, "correct": 0}},
        "examples": [],
    }

    model.eval()

    with torch.no_grad():
        for i, ex in enumerate(examples):
            if (i + 1) % 50 == 0:
                acc_so_far = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
                print(f"  {i+1}/{len(examples)}... (running acc: {acc_so_far:.1f}%)")

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
