#!/usr/bin/env python3
"""
Phase 1: Regenerate benchmark with improved distractors.

This script regenerates the question set using the updated distractor builders:
  - compound_aggressor_victim: local bystander pool + role reversal
  - compound_aggressor_action_victim: local bystander injection + role reversal
  - sequence_verification: local bystander swaps

Output: generated_questions_plan2.json (same structure as original, ready for SFT)
"""

import json
import random
from pathlib import Path

from prompt_generator.generator import QuestionGenerator
from prompt_generator.answer_bank import normalize_entry


def regenerate_benchmark(
    annotations_path: str = "annotations.json",
    output_path: str = "plan2_data/generated_questions_plan2.json",
    num_distractors: int = 7,
    trick_probability: float = 0.1,
    seed: int = 42,
) -> dict:
    """Regenerate the full benchmark using improved distractor builders."""
    random.seed(seed)

    # Load annotations
    with open(annotations_path) as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations")

    # Initialize generator
    generator = QuestionGenerator(
        annotations=annotations,
        num_distractors=num_distractors,
        trick_probability=trick_probability,
    )

    print(f"Generating questions (target: 5-6 per video)...")

    questions_by_video = {}
    generated_count = 0
    failed_count = 0

    for i, entry in enumerate(annotations):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(annotations)} videos processed...")

        video_name = entry.get("video_name", f"video_{i}")
        video_questions = []

        # Generate primary questions
        for _ in range(10):  # attempt 10 times to get 5-6 primary questions
            q = generator.generate_question(entry=entry)
            if q and not q.is_secondary:
                video_questions.append({
                    "video_name": q.video_name,
                    "question_type": q.question_type,
                    "is_secondary": False,
                    "is_trick": q.is_trick,
                    "prompt": q.prompt,
                    "answers": q.answers,
                    "correct_answer": q.correct_answer,
                    "correct_index": q.correct_index,
                })
                generated_count += 1

                if len(video_questions) >= 6:
                    break

        if video_questions:
            questions_by_video[video_name] = video_questions
        else:
            failed_count += 1

    print(f"Generated {generated_count} primary questions from {len(questions_by_video)} videos")
    print(f"Failed to generate questions for {failed_count} videos")

    # Prepare output
    output_data = {
        "total_annotations": len(annotations),
        "questions_generated": generated_count,
        "videos_with_questions": len(questions_by_video),
        "num_distractors": num_distractors,
        "trick_probability": trick_probability,
        "seed": seed,
        "questions_by_video": questions_by_video,
    }

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Wrote {len(questions_by_video)} videos to {output_path}")
    return output_data


if __name__ == "__main__":
    regenerate_benchmark()
