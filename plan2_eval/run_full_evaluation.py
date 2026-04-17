#!/usr/bin/env python3
"""
Phase 6: Full Evaluation & Ablation Study.

Evaluates all pipeline stages on the language backbone:
  1. Phase 1 SFT baseline
  2. Phase 3 SFT+CoT
  3. Phase 5 ADPO final
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from collections import defaultdict


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


def load_model(model_path: str, model_name: str = "OpenGVLab/InternVL2_5-8B"):
    """Load language backbone + LoRA checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = full_model.language_model

    adapter_path = _find_adapter(model_path)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, device_map="auto")
        print(f"  Loaded LoRA checkpoint from {adapter_path}")
    else:
        print(f"  WARNING: No adapter found at {model_path} — evaluating base model")

    model.eval()
    return tokenizer, model


def evaluate_model(
    model_path: str,
    test_data_path: str,
    model_name: str = "OpenGVLab/InternVL2_5-8B",
) -> Dict[str, Any]:
    """Evaluate a model checkpoint on test set."""
    stage = Path(model_path).name
    print(f"\n{'='*80}")
    print(f"Evaluating: {stage}")
    print(f"{'='*80}")

    tokenizer, model = load_model(model_path, model_name)

    with open(test_data_path) as f:
        examples = json.load(f)
    print(f"  {len(examples)} test examples")

    correct = 0
    total = 0
    results_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_trick = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for i, ex in enumerate(examples):
            if (i + 1) % 100 == 0:
                acc = correct / total * 100 if total > 0 else 0
                print(f"  [{i+1}/{len(examples)}] running acc: {acc:.1f}%")

            prompt = f"{ex['video_context']}\n\nQuestion: {ex['prompt']}\n\nAnswer:"
            correct_answer = ex["correct_answer"]

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            is_correct = (
                response.lower() == correct_answer.lower()
                or correct_answer.lower() in response.lower()
            )

            total += 1
            if is_correct:
                correct += 1

            qtype = ex["question_type"]
            results_by_type[qtype]["total"] += 1
            if is_correct:
                results_by_type[qtype]["correct"] += 1

            trick_key = "trick" if ex.get("is_trick", False) else "normal"
            results_by_trick[trick_key]["total"] += 1
            if is_correct:
                results_by_trick[trick_key]["correct"] += 1

    overall_acc = correct / total * 100 if total > 0 else 0
    print(f"\n  Overall: {overall_acc:.1f}% ({correct}/{total})")

    print("  By Question Type:")
    for qtype in sorted(results_by_type.keys()):
        stats = results_by_type[qtype]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {qtype:40s}: {acc:5.1f}% ({stats['correct']:3d}/{stats['total']:3d})")

    return {
        "overall_accuracy": overall_acc,
        "by_question_type": dict(results_by_type),
        "by_trick": dict(results_by_trick),
        "total_samples": total,
    }


def compare_models(
    sft_path: str = "plan2_models/sft_baseline",
    cot_path: str = "plan2_models/cot_sft",
    adpo_path: str = "plan2_models/adpo_final",
    test_data: str = "plan2_data/sft_test.json",
    output_path: str = "plan2_eval/full_evaluation_results.json",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
):
    """Compare all three pipeline stages."""
    print("=" * 80)
    print("PHASE 6: FULL EVALUATION & ABLATION STUDY")
    print("=" * 80)

    results = {}
    for label, path in [("phase1_sft", sft_path), ("phase3_cot_sft", cot_path), ("phase5_adpo", adpo_path)]:
        if Path(path).exists():
            results[label] = evaluate_model(path, test_data, model_name)
        else:
            print(f"Warning: {path} not found, skipping")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("Ablation Summary:")
    print(f"{'Stage':<25} {'Overall Acc':<15} {'Trick Acc':<15}")
    print("-" * 55)
    for stage_name in ["phase1_sft", "phase3_cot_sft", "phase5_adpo"]:
        if stage_name not in results:
            continue
        acc = results[stage_name]["overall_accuracy"]
        trick_stats = results[stage_name]["by_trick"]
        trick_acc = (
            trick_stats["trick"]["correct"] / trick_stats["trick"]["total"] * 100
            if "trick" in trick_stats and trick_stats["trick"]["total"] > 0
            else 0
        )
        print(f"{stage_name:<25} {acc:>6.1f}%{'':<8} {trick_acc:>6.1f}%")

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default="plan2_models/sft_baseline")
    parser.add_argument("--cot", default="plan2_models/cot_sft")
    parser.add_argument("--adpo", default="plan2_models/adpo_final")
    parser.add_argument("--test-data", default="plan2_data/sft_test.json")
    parser.add_argument("--output", default="plan2_eval/full_evaluation_results.json")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL2_5-8B")
    args = parser.parse_args()

    compare_models(
        sft_path=args.sft,
        cot_path=args.cot,
        adpo_path=args.adpo,
        test_data=args.test_data,
        output_path=args.output,
        model_name=args.model_name,
    )
