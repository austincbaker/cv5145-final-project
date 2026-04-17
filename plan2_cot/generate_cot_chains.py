#!/usr/bin/env python3
"""
Phase 2: Generate CoT reasoning chains via local InternVL2.5-8B teacher.

For compound/complex question types, uses a local InternVL2.5-8B model to
generate step-by-step reasoning that leads to the correct answer. Runs entirely
on local GPU hardware - no API costs.

Supports incremental checkpointing for SLURM requeue survival.

Input: SFT training data split (from Phase 1)
Output: JSON file with CoT chains merged into training examples
"""

import json
import sys
import argparse
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModel

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
    max_new_tokens: int = 300,
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


def _example_key(ex: dict) -> str:
    """Unique key for an example to track what's been processed."""
    return f"{ex['video_name']}|{ex['question_type']}|{ex['question_index']}"


def generate_cot_data(
    train_data_path: str = "plan2_data/sft_train.json",
    output_path: str = "plan2_cot/cot_chains_train.json",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    sample_rate: float = 1.0,
    dry_run: bool = False,
):
    """Generate CoT chains for training data using local teacher model."""
    checkpoint_path = output_path + ".checkpoint"

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

    # Resume from checkpoint if available
    cot_results = []
    processed_keys = set()
    failed_count = 0
    filtered_count = 0

    if Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        cot_results = checkpoint_data.get("cot_results", [])
        failed_count = checkpoint_data.get("failed_count", 0)
        filtered_count = checkpoint_data.get("filtered_count", 0)
        processed_keys = {_example_key(r) for r in cot_results}
        # Also count filtered/failed as processed
        for k in checkpoint_data.get("processed_keys", []):
            processed_keys.add(k)
        print(f"  Resuming from checkpoint: {len(processed_keys)} already processed, {len(cot_results)} chains saved")

    remaining = [ex for ex in eligible if _example_key(ex) not in processed_keys]
    print(f"\nGenerating CoT chains for {len(remaining)} remaining examples (of {len(eligible)} total)...")

    if not remaining:
        print("All examples already processed!")
    elif not dry_run:
        tokenizer, model = load_teacher_model(model_name)
    else:
        tokenizer, model = None, None
        print("[DRY RUN] Skipping model load")

    start_time = time.time()
    for i, example in enumerate(remaining):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (len(remaining) - i - 1) / rate / 60 if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(remaining)}] {len(cot_results)} chains "
                f"({filtered_count} filtered, {failed_count} failed) "
                f"- {rate:.2f}/s, ETA {eta_min:.0f}min"
            )

        key = _example_key(example)

        if dry_run:
            chain = "[DRY RUN] Sample reasoning chain for pipeline testing"
        else:
            chain = generate_cot_chain(tokenizer, model, example)

        processed_keys.add(key)

        if chain is None:
            failed_count += 1
        elif not filter_cot_chain(chain, example["correct_answer"]):
            filtered_count += 1
        else:
            result = example.copy()
            result["reasoning_chain"] = chain
            result["used_cot"] = True
            cot_results.append(result)

        # Save checkpoint every 100 examples
        if (i + 1) % 100 == 0:
            _save_checkpoint(checkpoint_path, cot_results, processed_keys, failed_count, filtered_count)
            print(f"  [checkpoint saved: {len(cot_results)} chains, {len(processed_keys)} processed]")

    # Final checkpoint
    _save_checkpoint(checkpoint_path, cot_results, processed_keys, failed_count, filtered_count)

    # Add non-CoT examples
    for example in simple:
        result = example.copy()
        result["used_cot"] = False
        cot_results.append(result)

    for example in other:
        result = example.copy()
        result["used_cot"] = False
        cot_results.append(result)

    print(f"\nGeneration complete:")
    print(f"  CoT chains: {sum(1 for r in cot_results if r.get('used_cot'))}")
    print(f"  Failed: {failed_count}")
    print(f"  Filtered (low quality): {filtered_count}")
    print(f"  Simple/other (direct): {len(simple) + len(other)}")
    print(f"  Total output examples: {len(cot_results)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cot_results, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Clean up checkpoint
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print("Checkpoint cleaned up")

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


def _save_checkpoint(path, cot_results, processed_keys, failed_count, filtered_count):
    """Save incremental checkpoint."""
    data = {
        "cot_results": cot_results,
        "processed_keys": list(processed_keys),
        "failed_count": failed_count,
        "filtered_count": filtered_count,
    }
    with open(path, "w") as f:
        json.dump(data, f)


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
