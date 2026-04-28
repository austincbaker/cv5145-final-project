#!/usr/bin/env python3
"""
Generate counting + social appropriateness questions on a 10% video sample.

Temporarily enables social_appropriateness and generates only counting
and social appropriateness question types.

Usage:
    python freeform_eval/generate_counting_social.py \
        --annotations annotations.json \
        --sample-frac 0.10 \
        --seed 42 \
        -o freeform_eval/counting_social_questions.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from prompt_generator.templates import (
    QUESTION_CATEGORIES,
    QUESTION_TEMPLATES,
    QuestionCategory,
    QuestionType,
)
from prompt_generator.generator import QuestionGenerator


TARGET_TYPES = {
    QuestionType.ROLE_COUNT_AGGRESSOR,
    QuestionType.ROLE_COUNT_VICTIM,
    QuestionType.ROLE_COUNT_BYSTANDER,
    QuestionType.COMPOUND_AGGRESSOR_VICTIM_COUNT,
    QuestionType.COMPOUND_VICTIM_BYSTANDER_COUNT,
    QuestionType.SOCIAL_APPROPRIATENESS,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="freeform_eval/counting_social_questions.json")
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as f:
        anns = json.load(f)

    random.seed(args.seed)
    sample_size = max(1, int(len(anns) * args.sample_frac))
    sample = random.sample(anns, sample_size)
    print(f"Sampled {len(sample)} / {len(anns)} videos")

    # Temporarily enable social_appropriateness
    original_categories = dict(QUESTION_CATEGORIES)
    QUESTION_CATEGORIES[QuestionType.SOCIAL_APPROPRIATENESS] = QuestionCategory.COMPOUND

    gen = QuestionGenerator(sample, num_distractors=7, trick_probability=0.0)
    all_questions = gen.generate_all_questions()

    # Restore
    QUESTION_CATEGORIES.clear()
    QUESTION_CATEGORIES.update(original_categories)

    # Filter to only target types
    filtered = [q for q in all_questions if QuestionType(q.question_type) in TARGET_TYPES]
    print(f"Generated {len(all_questions)} total, kept {len(filtered)} (counting + social)")

    by_type = defaultdict(int)
    for q in filtered:
        by_type[q.question_type] += 1
    for qt, n in sorted(by_type.items()):
        print(f"  {qt}: {n}")

    # Build output
    questions_by_video = defaultdict(list)
    for q in filtered:
        questions_by_video[q.video_name].append({
            "video_name": q.video_name,
            "question_type": q.question_type,
            "prompt": q.prompt,
            "answers": q.answers,
            "correct_answer": q.correct_answer,
            "correct_index": q.correct_index,
            "is_trick": q.is_trick,
            "option_hardness": q.option_hardness,
        })

    output = {
        "metadata": {
            "num_videos": len(questions_by_video),
            "num_questions": len(filtered),
            "sample_fraction": args.sample_frac,
            "seed": args.seed,
            "question_types": sorted(by_type.keys()),
        },
        "questions_by_video": dict(questions_by_video),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Show a few social appropriateness examples
    social = [q for q in filtered if q.question_type == "social_appropriateness"]
    if social:
        print(f"\nSample social appropriateness questions:")
        for q in social[:3]:
            print(f"  {q.video_name}: {q.prompt}")
            for i, a in enumerate(q.answers):
                marker = "+" if i == q.correct_index else " "
                print(f"    {marker} {i+1}. {a}")
            print()

    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
