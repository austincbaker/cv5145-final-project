#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from .model_loader import ModelConfig
from .video_processor import VideoProcessorConfig
from .evaluator import VideoQuestionEvaluator, save_evaluation_results


def load_annotations(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "annotations" in data:
            return data["annotations"]
        return [data]
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")


def progress_printer(current: int, total: int, result) -> None:
    status = "CORRECT" if result.is_correct else "WRONG"
    if result.error:
        status = f"ERROR: {result.error}"

    print(f"[{current}/{total}] {result.video_name}")
    print(f"  Type: {result.question_type}")
    print(f"  Question: {result.prompt[:80]}...")
    print(f"  Answer Options:")
    for i, answer in enumerate(result.answers):
        marker = "✓" if i == result.correct_index else " "
        selected = "←" if i == result.model_selected_index else ""
        print(f"    {marker} {i + 1}. {answer} {selected}")
    print(f"  Status: {status}")
    print()


def checkpoint_progress_printer(
    current: int, 
    total: int, 
    result, 
    video_name=None, 
    question_num=None, 
    total_questions=None,
    status=None
) -> None:
    """Progress printer for checkpoint mode."""
    if status == "starting":
        print(f"\n[{current}/{total}] {video_name} - Processing questions...")
    elif status == "completed":
        print(f"  ✓ Completed - Checkpoint saved")
    elif result is not None:
        # Individual question progress
        status_symbol = "✓" if result.is_correct else "✗"
        if question_num == 1:
            print(f"  ", end="")
        print(status_symbol, end="")
        if question_num % 10 == 0 or question_num == total_questions:
            print(f" [{question_num}/{total_questions}]")
        sys.stdout.flush()


def run_evaluation(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate video model on generated questions"
    )
    parser.add_argument(
        "annotations_json",
        help="Path to JSON file containing video annotations",
    )
    parser.add_argument(
        "video_dir",
        help="Directory containing video files",
    )
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=10,
        help="Number of questions to evaluate (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory for output JSON file (default: current directory)",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        default="AIDC-AI/Ovis2.5-2B",
        help="HuggingFace model path (default: AIDC-AI/Ovis2.5-2B)",
    )
    parser.add_argument(
        "-f",
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to extract from each video (default: 8)",
    )
    parser.add_argument(
        "-d",
        "--num-distractors",
        type=int,
        default=5,
        help="Number of distractor answers per question (default: 5, giving 6 total options)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=512,
        help="Thinking budget for model (default: 512)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens for generation (default: 1024)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image resize dimension (default: 224)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check video/annotation matching, don't run evaluation",
    )
    parser.add_argument(
        "--all-questions",
        action="store_true",
        help="Generate all question types for all videos (ignores -n)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh evaluation, ignoring existing checkpoint",
    )

    parsed = parser.parse_args(args)

    try:
        annotations = load_annotations(parsed.annotations_json)
    except FileNotFoundError:
        print(f"Error: File not found: {parsed.annotations_json}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    video_dir = Path(parsed.video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}", file=sys.stderr)
        sys.exit(1)

    model_config = ModelConfig(
        model_path=parsed.model_path,
        thinking_budget=parsed.thinking_budget,
        max_new_tokens=parsed.max_new_tokens,
        image_size=(parsed.image_size, parsed.image_size),
    )

    video_config = VideoProcessorConfig(num_frames=parsed.num_frames)

    print(f"Loading model: {parsed.model_path}")
    print(f"Video directory: {video_dir}")
    print(f"Total annotations: {len(annotations)} entries")
    print(f"Frames per video: {parsed.num_frames}")
    if parsed.all_questions:
        print("Mode: ALL QUESTIONS on ALL VIDEOS")
    else:
        print(f"Questions to evaluate: {parsed.num_questions}")
    print()
    
    print("Scanning video directory and matching with annotations...")

    evaluator = VideoQuestionEvaluator(
        annotations=annotations,
        video_dir=video_dir,
        model_config=model_config,
        video_config=video_config,
        num_distractors=parsed.num_distractors,
    )
    
    print()
    
    # If check-only mode, show stats and exit
    if parsed.check_only:
        stats = evaluator.get_availability_stats()
        print("=" * 60)
        print("VIDEO AVAILABILITY CHECK")
        print("=" * 60)
        print(f"Total videos in directory: {stats['total_videos_in_dir']}")
        print(f"Total annotations provided: {len(annotations)}")
        print(f"Matched annotations with videos: {stats['matched']}")
        print(f"\nReady to evaluate {stats['matched']} video(s)")
        print("\nTo run evaluation, remove --check-only flag")
        return

    print("Loading model (this may take a moment)...")
    evaluator.load_model()
    print("Model loaded successfully.")
    print()

    progress_cb = None if parsed.quiet else progress_printer

    try:
        if parsed.all_questions:
            # Use checkpoint-based evaluation for all questions
            output_dir = Path(parsed.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = output_dir / f"evaluation_{timestamp}.checkpoint.jsonl"
            output_path = output_dir / f"evaluation_{timestamp}.json"
            
            # Use checkpoint progress printer
            checkpoint_cb = None if parsed.quiet else checkpoint_progress_printer
            
            print(f"Checkpoint file: {checkpoint_path}")
            print(f"Output file: {output_path}")
            print()
            
            results = evaluator.evaluate_all_with_checkpoint(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                resume=not parsed.no_resume,
                max_retries=3,
                progress_callback=checkpoint_cb,
            )
            
            # Print summary from results dict
            print()
            print("=" * 60)
            print("EVALUATION SUMMARY")
            print("=" * 60)
            print(f"Model: {results['model_path']}")
            print(f"Total videos evaluated: {results['total_videos_evaluated']}")
            print(f"Total questions: {results['total_questions']}")
            print(f"Correct: {results['correct_count']}")
            print(f"Overall accuracy: {results['accuracy']:.2%}")
            print()
            print("Accuracy by question type:")
            for qtype, stats in sorted(results['accuracy_by_type'].items()):
                print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
            print()
            print(f"Results saved to: {output_path}")
            print(f"Checkpoint saved to: {checkpoint_path}")
            
            return
            
        else:
            # Generate random sample of questions
            summary = evaluator.evaluate_random(
                num_questions=parsed.num_questions,
                progress_callback=progress_cb,
            )
    finally:
        evaluator.unload_model()

    output_file = save_evaluation_results(summary, parsed.output_dir)

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {summary.model_path}")
    print(f"Total questions: {summary.total_questions}")
    print(f"Correct: {summary.correct_count}")
    print(f"Overall accuracy: {summary.accuracy:.2%}")
    print()
    print("Accuracy by question type:")
    for qtype, stats in sorted(summary.accuracy_by_type.items()):
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
    print()
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()