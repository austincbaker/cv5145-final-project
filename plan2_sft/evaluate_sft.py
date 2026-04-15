#!/usr/bin/env python3
"""
Phase 1: Evaluate SFT Baseline.

Measure accuracy on test/val set:
  - Overall accuracy
  - Per-question-type breakdown
  - Trick question performance
  - Confidence on correct vs incorrect
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import torch
from transformers import AutoProcessor, AutoModel
from peft import PeftModel


def evaluate_sft(
    model_path: str,
    test_data: str,
    processor=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate SFT model on test set."""
    # Load model
    print(f"Loading model from {model_path}")
    base_model_name = "OpenGVLab/InternVL2_5-8B"
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA weights if available
    if Path(model_path).exists():
        try:
            model = PeftModel.from_pretrained(model, model_path, device_map="auto")
            print(f"Loaded LoRA weights from {model_path}")
        except Exception as e:
            print(f"Warning: could not load LoRA weights: {e}")

    # Load processor
    if processor is None:
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

    # Load test data
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
                print(f"  {i+1}/{len(examples)}...")

            prompt = f"{ex['video_context']}\n\nQuestion: {ex['prompt']}\n\nAnswer:"
            correct_answer = ex["correct_answer"]

            # Tokenize input
            inputs = processor(prompt, return_tensors="pt").to(device)

            # Generate response (limited length for speed)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=1,
                    temperature=0.7,
                )

            # Decode
            response = processor.decode(outputs[0], skip_special_tokens=True)

            # Extract answer (everything after "Answer:")
            if "Answer:" in response:
                predicted_answer = response.split("Answer:")[-1].strip()
            else:
                predicted_answer = response.strip()

            # Check if correct (exact match or substring match)
            is_correct = (
                predicted_answer.lower() == correct_answer.lower()
                or predicted_answer.lower() in [ans.lower() for ans in ex["all_answers"]]
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
                "prompt": ex["prompt"],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
            })

    # Print results
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
    parser.add_argument("--model", default="plan2_models/sft_baseline", help="Path to trained model")
    parser.add_argument("--test-data", default="plan2_data/sft_test.json", help="Test data path")
    parser.add_argument("--output", default="plan2_eval/sft_baseline_results.json", help="Output results path")
    args = parser.parse_args()

    results = evaluate_sft(args.model, args.test_data)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        # Remove examples from serialization for brevity
        results_to_save = {k: v for k, v in results.items() if k != "examples"}
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")
