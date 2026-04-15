#!/usr/bin/env python3
"""
Phase 4: Extract ADPO Preference Pairs.

Extracts preference pairs from the improved distractor pipeline:
  Chosen: correct answer (with CoT for compounds if available)
  Rejected: hard negatives ranked by hardness type

Hardness priority:
  1. Role reversals (3.2x improvement expected)
  2. Bystander substitutions
  3. Wrong action, correct roles
  4. Other in-cast distractors
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter


def extract_preference_pairs(
    questions_path: str = "plan2_data/generated_questions_plan2.json",
    cot_chains_path: str = "plan2_cot/cot_chains_train.json",
    output_path: str = "plan2_adpo/preference_pairs_train.json",
    num_rejected_per_chosen: int = 3,
    seed: int = 42,
):
    """Extract preference pairs for ADPO training."""
    random.seed(seed)

    # Load questions
    with open(questions_path) as f:
        questions_data = json.load(f)
    questions_by_video = questions_data["questions_by_video"]

    # Load CoT chains (optional, for enriching chosen responses)
    cot_by_video_qidx = {}
    if Path(cot_chains_path).exists():
        with open(cot_chains_path) as f:
            cot_examples = json.load(f)
        for ex in cot_examples:
            key = (ex["video_name"], ex.get("question_index", 0))
            if ex.get("used_cot") and "reasoning_chain" in ex:
                cot_by_video_qidx[key] = ex["reasoning_chain"]
        print(f"Loaded {len(cot_by_video_qidx)} CoT chains")
    else:
        print("No CoT chains found; preference pairs will use direct answers")

    # Classify distractors by hardness
    def classify_distractor(distractor: str, correct_answer: str) -> str:
        """Classify distractor hardness (quick heuristic)."""
        d_lower = distractor.lower()
        c_lower = correct_answer.lower()

        # Role reversal: aggressor/victim swapped
        if "(aggressor)" in d_lower and "(victim)" in d_lower:
            return "role_reversal"
        # Bystander: typically in-cast people
        if "bystander" in d_lower:
            return "bystander_substitution"
        # Wrong action: different action mentioned
        if "did" in d_lower and "did" in c_lower:
            return "wrong_action"
        # Cross-video
        return "cross_video"

    # Extract pairs
    preference_pairs = []
    pair_stats = Counter()

    for video_name, questions in questions_by_video.items():
        for q_idx, q in enumerate(questions):
            if q.get("is_secondary") or q.get("is_trick"):
                continue

            qtype = q["question_type"]
            correct_answer = q["correct_answer"]
            answers = q["answers"]
            correct_index = q.get("correct_index", -1)

            # Chosen: correct answer, optionally with CoT
            key = (video_name, q_idx)
            reasoning_chain = cot_by_video_qidx.get(key)

            chosen = {
                "answer": correct_answer,
                "reasoning": reasoning_chain if reasoning_chain else None,
            }

            # Rejected: distractors, classified and ranked
            distractors = [
                a for i, a in enumerate(answers)
                if i != correct_index and a != correct_answer
            ]

            # Classify and rank
            classified = []
            for distractor in distractors:
                hardness = classify_distractor(distractor, correct_answer)
                if hardness != "cross_video":  # Prioritize in-cast
                    classified.append((distractor, hardness))

            # Sort by priority: role_reversal > bystander > wrong_action > other
            priority = {
                "role_reversal": 0,
                "bystander_substitution": 1,
                "wrong_action": 2,
                "other": 3,
            }
            classified.sort(key=lambda x: priority.get(x[1], 99))

            # Add rejected responses
            rejected = []
            for distractor, hardness in classified[:num_rejected_per_chosen]:
                rejected.append({
                    "answer": distractor,
                    "hardness": hardness,
                })
                pair_stats[hardness] += 1

            # Skip if no good distractors
            if not rejected:
                continue

            # Create preference pair
            pair = {
                "video_name": video_name,
                "question_type": qtype,
                "prompt": q["prompt"],
                "video_context": "",  # Will be populated if needed
                "chosen": chosen,
                "rejected": rejected,
            }

            preference_pairs.append(pair)

    print(f"\nExtracted {len(preference_pairs)} preference pairs")
    print("Hardness distribution:")
    for hardness, count in sorted(pair_stats.items(), key=lambda x: -x[1]):
        print(f"  {hardness:30s}: {count:6,}")

    # Save pairs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(preference_pairs, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary
    print(f"\nPreference pair statistics:")
    print(f"  Total pairs: {len(preference_pairs):,}")
    print(f"  Avg rejected per pair: {sum(len(p['rejected']) for p in preference_pairs) / len(preference_pairs):.1f}")
    print(f"  With CoT reasoning: {sum(1 for p in preference_pairs if p['chosen']['reasoning'])}")

    return preference_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="plan2_data/generated_questions_plan2.json")
    parser.add_argument("--cot-chains", default="plan2_cot/cot_chains_train.json")
    parser.add_argument("--output", default="plan2_adpo/preference_pairs_train.json")
    parser.add_argument("--num-rejected", type=int, default=3)
    args = parser.parse_args()

    extract_preference_pairs(
        questions_path=args.questions,
        cot_chains_path=args.cot_chains,
        output_path=args.output,
        num_rejected_per_chosen=args.num_rejected,
    )
