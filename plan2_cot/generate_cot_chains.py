#!/usr/bin/env python3
"""
Phase 2: Generate CoT reasoning chains via local InternVL2.5-8B teacher.

For compound/complex question types, uses a local InternVL2.5-8B model to
generate step-by-step reasoning that leads to the correct answer. Runs entirely
on local GPU hardware -no API costs.

Input: SFT training data split (from Phase 1)
Output: JSON file with CoT chains merged into training examples
"""

import json
import argparse
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel


# Question types eligible for CoT (compounds and complex)
COT_ELIGIBLE_TYPES = {
    "compound_aggressor_victim",
    "compound_aggressor_action_victim",
    "compound_action_victims",
    "compound_aggressor_location",
    "compound_action_location",
    "sequence_verification",
    "aggressor_identification",
    "victim_recognition",
}

SIMPLE_TYPES = {
    "primary_action",
}


def build_cot_prompt(example: dict) -> str:
    """Build a prompt for the teacher model to generate a reasoning chain."""
    context = example["video_context"]
    question = example["prompt"]
    answer = example["correct_answer"]

    prompt = f"""You are analyzing a video of an aggressive interaction. Generate a step-by-step reasoning chain that leads to the correct answer.

Video Context:
{context}

Question: {question}

Correct Answer: {answer}

Provide your reasoning in the following format:
1. Identify key people and their roles in the video
2. Describe the actions taking place
3. Note any relevant environmental details
4. Determine the answer step-by-step
5. Final answer

Keep each step concise and grounded in the video context provided."""

    return prompt


def load_teacher_model(model_name: str = "OpenGVLab/InternVL2_5-8B"):
    """Load InternVL2.5-8B teacher model for CoT generation."""
    print(f"Loading teacher model: {model_name}")
    print("(Will download ~52GB on first run if not already cached)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    # Print memory footprint
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[OK] Teacher model loaded ({allocated:.1f} GB GPU memory)")
    else:
        print("[OK] Teacher model loaded (CPU mode)")

    return tokenizer, model


def generate_cot_chain(
    tokenizer,
    model,
    example: dict,
    max_new_tokens: int = 500,
) -> Optional[str]:
    """Generate a CoT reasoning chain using the local teacher model."""
    prompt = build_cot_prompt(example)

    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    try:
        # InternVL2.5 chat interface: pixel_values=None for text-only mode
        response = model.chat(
            tokenizer,
            None,
            prompt,
            generation_config,
        )
        return response.strip() if response else None
    except Exception as e:
        print(f"  Error generating chain: {e}")
        return None


def filter_cot_chain(chain: str, correct_answer: str) -> bool:
    """Check if CoT chain is high quality (reaches correct answer)."""
    if not chain or len(chain) < 50:
        return False

    chain_lower = chain.lower()
    answer_lower = correct_answer.lower()

    tokens = answer_lower.split()[:3]
    found_count = sum(1 for token in tokens if token in chain_lower)

    return found_count >= 2


def generate_cot_data(
    train_data_path: str = "plan2_data/sft_train.json",
    output_path: str = "plan2_cot/cot_chains_train.json",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    sample_rate: float = 1.0,
    dry_run: bool = False,
):
    """Generate CoT chains for training data using local teacher model."""
    # Load training data
    with open(train_data_path) as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} training examples")

    eligible = [ex for ex in examples if ex["question_type"] in COT_ELIGIBLE_TYPES]
    simple = [ex for ex in examples if ex["question_type"] in SIMPLE_TYPES]
    other = [
        ex for ex in examples
        if ex["question_type"] not in COT_ELIGIBLE_TYPES
        and ex["question_type"] not in SIMPLE_TYPES
    ]

    print(f"  CoT-eligible (compounds/complex): {len(eligible)}")
    print(f"  Simple (direct only): {len(simple)}")
    print(f"  Other: {len(other)}")

    if sample_rate < 1.0:
        import random
        random.seed(42)
        eligible = random.sample(eligible, int(len(eligible) * sample_rate))
        print(f"  Sampled to {len(eligible)} for CoT generation")

    # Load teacher model (unless dry run)
    if not dry_run:
        tokenizer, model = load_teacher_model(model_name)
    else:
        tokenizer, model = None, None
        print("[DRY RUN] Skipping model load")

    cot_results = []
    failed_count = 0
    filtered_count = 0

    print(f"\nGenerating CoT chains for {len(eligible)} examples...")
    print()

    start_time = time.time()
    for i, example in enumerate(eligible):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (len(eligible) - i - 1) / rate / 60 if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(eligible)}] Generated {len(cot_results)} chains "
                f"({filtered_count} filtered) -{rate:.2f}/s, ETA {eta_min:.0f}min"
            )

        if dry_run:
            chain = "[DRY RUN] Sample reasoning chain for pipeline testing"
        else:
            chain = generate_cot_chain(tokenizer, model, example)

        if chain is None:
            failed_count += 1
            continue

        if not filter_cot_chain(chain, example["correct_answer"]):
            filtered_count += 1
            continue

        result = example.copy()
        result["reasoning_chain"] = chain
        result["used_cot"] = True
        cot_results.append(result)

    # Add non-CoT examples without reasoning chains
    for example in simple:
        result = example.copy()
        result["used_cot"] = False
        cot_results.append(result)

    for example in other:
        result = example.copy()
        result["used_cot"] = False
        cot_results.append(result)

    print(f"\nGeneration complete:")
    print(f"  Generated: {len(cot_results)} examples with metadata")
    print(f"  Failed: {failed_count}")
    print(f"  Filtered (low quality): {filtered_count}")
    print(f"  Simple (direct only): {len(simple)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cot_results, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Summary by type
    by_type = defaultdict(lambda: {"total": 0, "cot": 0})
    for ex in cot_results:
        qtype = ex["question_type"]
        by_type[qtype]["total"] += 1
        if ex.get("used_cot"):
            by_type[qtype]["cot"] += 1

    print("\nCoT coverage by question type:")
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        pct = stats["cot"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {qtype:40s}: {stats['cot']:4d}/{stats['total']:4d} ({pct:5.1f}%)")

    return cot_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="plan2_data/sft_train.json")
    parser.add_argument("--output", default="plan2_cot/cot_chains_train.json")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Sample subset (0-1) for faster iteration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't load model, just test pipeline")
    args = parser.parse_args()

    generate_cot_data(
        train_data_path=args.train_data,
        output_path=args.output,
        model_name=args.model_name,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run,
    )
