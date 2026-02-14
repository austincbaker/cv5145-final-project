"""
Pre-generate questions for all videos and save to a static JSON file.
This allows reusing the same questions across different models.
"""
import argparse
import json
from pathlib import Path
from .generator import QuestionGenerator
from .templates import QuestionType
from .evaluation.run_evaluation import load_annotations


# Selected question types for evaluation
SELECTED_QUESTION_TYPES = [
    # Simple
    QuestionType.AGGRESSOR_ID,
    QuestionType.VICTIM_RECOGNITION,
    # Complex
    QuestionType.COMPOUND_AGGRESSOR_VICTIM,
    QuestionType.COMPOUND_AGGRESSOR_ACTION_VICTIM,
    QuestionType.COMPOUND_ACTION_VICTIMS,
    # Summary
    QuestionType.INTERACTION_SUMMARY,
]


def generate_questions_for_all_videos(
    annotations_path: str,
    output_path: str,
    num_distractors: int = 7,
) -> dict:
    """
    Generate questions for all videos and save to JSON.

    Args:
        annotations_path: Path to annotations JSON file
        output_path: Path to save generated questions JSON
        num_distractors: Number of distractor answers per question

    Returns:
        Dictionary with generated questions and metadata
    """
    # Load annotations
    annotations = load_annotations(annotations_path)
    print(f"Loaded {len(annotations)} annotations")

    # Create generator
    generator = QuestionGenerator(annotations, num_distractors=num_distractors)

    # Generate questions for each video
    all_questions = []
    skipped_count = 0

    for entry in annotations:
        video_name = entry.get("video_name", "unknown")

        # Generate only selected question types for this video
        for question_type in SELECTED_QUESTION_TYPES:
            question = generator.generate_question(
                entry=entry,
                question_type=question_type
            )

            if question is not None:
                # Convert to dict for JSON serialization
                question_dict = {
                    "video_name": question.video_name,
                    "question_type": question.question_type,
                    "prompt": question.prompt,
                    "answers": question.answers,
                    "correct_answer": question.correct_answer,
                    "correct_index": question.correct_index,
                }
                all_questions.append(question_dict)
            else:
                skipped_count += 1

    # Group questions by video for easier processing
    questions_by_video = {}
    for q in all_questions:
        video = q["video_name"]
        if video not in questions_by_video:
            questions_by_video[video] = []
        questions_by_video[video].append(q)

    # Calculate statistics
    question_counts = {}
    for q in all_questions:
        qtype = q["question_type"]
        question_counts[qtype] = question_counts.get(qtype, 0) + 1

    # Save to JSON
    output = {
        "metadata": {
            "num_videos": len(questions_by_video),
            "num_questions": len(all_questions),
            "num_distractors": num_distractors,
            "question_types": [qt.value for qt in SELECTED_QUESTION_TYPES],
            "question_counts_by_type": question_counts,
            "skipped_questions": skipped_count,
        },
        "questions_by_video": questions_by_video,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated questions:")
    print(f"  Videos: {len(questions_by_video)}")
    print(f"  Total questions: {len(all_questions)}")
    print(f"  Skipped questions: {skipped_count}")
    print(f"  Questions per video: {len(all_questions) / len(questions_by_video):.1f}")
    print(f"\nQuestions by type:")
    for qtype, count in sorted(question_counts.items()):
        print(f"  {qtype}: {count}")
    print(f"\nSaved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate questions for all videos"
    )
    parser.add_argument(
        "annotations_json",
        help="Path to JSON file containing video annotations",
    )
    parser.add_argument(
        "-o", "--output",
        default="generated_questions.json",
        help="Output JSON file path (default: generated_questions.json)",
    )
    parser.add_argument(
        "-d", "--num-distractors",
        type=int,
        default=7,
        help="Number of distractor answers per question (default: 7, giving 8 total options)",
    )

    args = parser.parse_args()

    try:
        generate_questions_for_all_videos(
            args.annotations_json,
            args.output,
            args.num_distractors,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
