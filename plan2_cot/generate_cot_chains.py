#!/usr/bin/env python3
"""
Phase 2: Generate CoT reasoning chains via GPT-4o.

For compound/complex question types, calls GPT-4o to generate step-by-step
reasoning that leads to the correct answer.

Input: SFT training data split (from Phase 1)
Output: JSON file with CoT chains merged into training examples
"""

import json
import os
from pathlib import Path
from typing import Optional
import time
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai")
    raise


# Question types eligible for CoT (compounds and complex)
COT_ELIGIBLE_TYPES = {
    "compound_aggressor_victim",
    "compound_aggressor_action_victim",
    "compound_action_victims",
    "compound_aggressor_location",
    "compound_action_location",
    "sequence_verification",
    "aggressor_identification",  # Can benefit from reasoning
    "victim_recognition",         # Can benefit from reasoning
}

SIMPLE_TYPES = {
    "primary_action",
}


def build_cot_prompt(example: dict) -> str:
    """Build a prompt for GPT-4o to generate reasoning chain."""
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


def generate_cot_chain(client: OpenAI, example: dict, max_retries: int = 3) -> Optional[str]:
    """Call GPT-4o to generate a CoT chain for an example."""
    prompt = build_cot_prompt(example)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.7,
                max_tokens=500,
            )
            chain = response.choices[0].message.content.strip()
            return chain
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries - 1} after error: {e}")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} retries: {e}")
                return None

    return None


def filter_cot_chain(chain: str, correct_answer: str) -> bool:
    """Check if CoT chain is high quality (reaches correct answer)."""
    if not chain or len(chain) < 50:
        return False

    # Check if answer appears in chain (heuristic)
    chain_lower = chain.lower()
    answer_lower = correct_answer.lower()

    # Look for key parts of the answer
    tokens = answer_lower.split()[:3]  # First 3 tokens of answer
    found_count = sum(1 for token in tokens if token in chain_lower)

    return found_count >= 2  # At least 2 key tokens must appear


def generate_cot_data(
    train_data_path: str = "plan2_data/sft_train.json",
    output_path: str = "plan2_cot/cot_chains_train.json",
    api_key: Optional[str] = None,
    sample_rate: float = 1.0,  # Can sample subset for cost management
    dry_run: bool = False,
):
    """Generate CoT chains for training data."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Set environment variable or pass api_key.")

    client = OpenAI(api_key=api_key)

    # Load training data
    with open(train_data_path) as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} training examples")

    # Filter to CoT-eligible types
    eligible = [ex for ex in examples if ex["question_type"] in COT_ELIGIBLE_TYPES]
    simple = [ex for ex in examples if ex["question_type"] in SIMPLE_TYPES]
    other = [ex for ex in examples if ex["question_type"] not in COT_ELIGIBLE_TYPES and ex["question_type"] not in SIMPLE_TYPES]

    print(f"  CoT-eligible (compounds/complex): {len(eligible)}")
    print(f"  Simple (direct only): {len(simple)}")
    print(f"  Other: {len(other)}")

    # Sample if needed
    if sample_rate < 1.0:
        import random
        eligible = random.sample(eligible, int(len(eligible) * sample_rate))
        print(f"  Sampled to {len(eligible)} for CoT generation")

    # Generate CoT chains
    cot_results = []
    failed_count = 0
    filtered_count = 0

    print(f"\nGenerating CoT chains for {len(eligible)} examples...")
    print("(This will take a while and incur OpenAI API costs)")
    print()

    for i, example in enumerate(eligible):
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(eligible)}] Generated {len(cot_results)} chains " f"({filtered_count} filtered out)")

        if dry_run:
            chain = "[DRY RUN] Sample reasoning chain"
        else:
            chain = generate_cot_chain(client, example)

        if chain is None:
            failed_count += 1
            continue

        # Quality filter
        if not filter_cot_chain(chain, example["correct_answer"]):
            filtered_count += 1
            continue

        # Add chain to example
        result = example.copy()
        result["reasoning_chain"] = chain
        result["used_cot"] = True
        cot_results.append(result)

    # Add simple examples without CoT
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

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cot_results, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Summary stats
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="plan2_data/sft_train.json")
    parser.add_argument("--output", default="plan2_cot/cot_chains_train.json")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sample subset (0-1) for cost management")
    parser.add_argument("--dry-run", action="store_true", help="Don't call API, just test pipeline")
    args = parser.parse_args()

    generate_cot_data(
        train_data_path=args.train_data,
        output_path=args.output,
        sample_rate=args.sample_rate,
        dry_run=args.dry_run,
    )
