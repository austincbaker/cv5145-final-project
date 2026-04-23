#!/usr/bin/env python3
"""
Text-only evaluation script.

Runs evaluation using only question text and answer options — no video frames.
Scoring, prompt format, and reported metrics mirror
train_model/eval/run_evaluation.py so a text-only run is directly comparable
to a normal (frame-based) run.

Usage:
    python text_only_eval.py \
        --questions-file generated_questions.json \
        --model qwen-7b \
        --output results_text_only/

    # Limit to first 50 videos for a quick test run
    python text_only_eval.py \
        --questions-file generated_questions.json \
        --model qwen-7b \
        --output results_text_only/ \
        --limit 50

Accepts either of:
  * generated_questions.json  (top-level {"metadata": ..., "questions_by_video": ...})
  * sft_test.json / cot chains (flat list of examples)

For the normal eval's primary-question set, run:
    python make_frameless_questions.py generated_questions.json
first and pass the _text_only.json to --questions-file.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


# Mirror of train_model/eval/run_evaluation.py::LETTER_RE so scoring is
# byte-identical between text-only and frame-based runs.
LETTER_RE = re.compile(r"\b([A-H])\b\s*[\)\.\:]?", re.IGNORECASE)


def parse_letter(resp: str) -> int | None:
    """Return 0..7 for an A..H letter in `resp`, or None if no letter found."""
    m = LETTER_RE.search(resp.strip())
    return (ord(m.group(1).upper()) - ord("A")) if m else None


@dataclass
class EvaluationResult:
    video_name: str
    question_type: str
    is_trick: bool
    prompt: str
    answers: list
    correct_answer: str
    correct_index: int
    model_response: str
    model_selected_index: int | None
    is_correct: bool
    error: str | None = None


def format_prompt(question: dict) -> str:
    """Letter-based MCQ prompt matching train_model/eval/run_evaluation.py
    minus the frame placeholders.
    """
    answers = question.get("answers") or question.get("all_answers") or []
    lines = [f"Question: {question['prompt']}"]
    if answers:
        lines.append("Options:")
        for i, opt in enumerate(answers):
            lines.append(f"{chr(ord('A') + i)}) {opt}")
        lines.append("")
        lines.append("Answer with the option letter (A, B, C, ...) followed by the option text.")
    return "\n".join(lines)


def score_response(response: str, answers: list, correct_index: int) -> tuple[int | None, bool, bool]:
    """Return (selected_index, is_correct, letter_parsed).

    Matches run_evaluation.py's scoring:
      1. If the response contains a standalone A..H letter, that letter is
         the model's answer (letter_parsed=True).
      2. Otherwise fall back to text-substring match on correct_answer.
    """
    resp_lower = response.strip().lower()
    correct_text = answers[correct_index].lower().strip() if 0 <= correct_index < len(answers) else ""

    parsed = parse_letter(response)
    if parsed is not None:
        return parsed, (parsed == correct_index), True
    if correct_text:
        return None, (correct_text in resp_lower), False
    return None, False, False


def _normalize_questions(
    questions_file: str,
) -> tuple[list[tuple[str, dict]], dict]:
    """Return a (video_name, question_dict) pair list plus source metadata.

    Accepts either schema:
      * {"metadata": ..., "questions_by_video": {vid: [q, ...]}}
      * [{"video_name": ..., ...}, ...]  (flat, as in sft_test.json)
    """
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions_by_video" in data:
        pairs = [(vid, q) for vid, qs in data["questions_by_video"].items() for q in qs]
        metadata = data.get("metadata", {})
    elif isinstance(data, list):
        # flat list — each example may use `all_answers` instead of `answers`
        pairs = [(ex.get("video_name", "unknown"), ex) for ex in data]
        metadata = {"source_schema": "flat_list"}
    else:
        raise ValueError(
            f"Unrecognised schema in {questions_file!r}: "
            f"expected top-level 'questions_by_video' dict or a flat list"
        )
    return pairs, metadata


def compute_summary(results: list[EvaluationResult], model_path: str) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total if total > 0 else 0.0
    letter_parsed = sum(1 for r in results if r.model_selected_index is not None)
    letter_parsed_rate = letter_parsed / total if total > 0 else 0.0

    by_type: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    by_trick: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_type[r.question_type]["total"] += 1
        if r.is_correct:
            by_type[r.question_type]["correct"] += 1
        trick_key = "trick" if r.is_trick else "normal"
        by_trick[trick_key]["total"] += 1
        if r.is_correct:
            by_trick[trick_key]["correct"] += 1

    accuracy_by_type = {
        qtype: {
            "total": c["total"],
            "correct": c["correct"],
            "accuracy": c["correct"] / c["total"] if c["total"] > 0 else 0.0,
        }
        for qtype, c in by_type.items()
    }
    accuracy_by_trick = {
        key: {
            "total": c["total"],
            "correct": c["correct"],
            "accuracy": c["correct"] / c["total"] if c["total"] > 0 else 0.0,
        }
        for key, c in by_trick.items()
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "mode": "text_only",
        "num_frames": 0,
        "total_questions": total,
        "correct_count": correct,
        "accuracy": accuracy,
        "letter_parsed_rate": letter_parsed_rate,
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_trick": accuracy_by_trick,
        "results": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Text-only model evaluation (no video frames)")
    parser.add_argument("--questions-file", required=True, help="Path to generated_questions.json or sft_test.json")
    parser.add_argument(
        "--model", required=True,
        help="Model shortcut (e.g. qwen-7b) or full HuggingFace path",
    )
    parser.add_argument("--output", default="results_text_only", help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N videos (useful for quick tests)",
    )
    args = parser.parse_args()

    # Make the package importable regardless of where the script is called from
    sys.path.insert(0, str(Path(__file__).parent))
    from prompt_generator.evaluation.model_loader import ModelConfig, create_loader
    from prompt_generator.evaluation.model_loader.registry import resolve_model_path

    model_path = resolve_model_path(args.model)
    print(f"Model: {model_path}")

    # Load questions. Accepts either generated_questions.json shape or a
    # flat list (sft_test.json). Grouping by video preserves the
    # "[v/N] video_name" progress line from the old script.
    pairs, metadata = _normalize_questions(args.questions_file)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for vid, q in pairs:
        grouped[vid].append(q)
    total_qs = sum(len(qs) for qs in grouped.values())
    print(f"Loaded {total_qs} questions across {len(grouped)} videos")

    if args.limit:
        video_names = sorted(grouped.keys())[: args.limit]
        grouped = {v: grouped[v] for v in video_names}
        print(f"  (Limited to first {args.limit} videos)")

    # Setup output directory and checkpoint
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.jsonl"

    # Resume support: skip videos already in the checkpoint
    completed_videos: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed_videos.add(json.loads(line).get("video_name", ""))
        if completed_videos:
            print(f"Resuming: {len(completed_videos)} videos already completed")

    # Load model (no video processor needed)
    model_config = ModelConfig(
        model_path=model_path,
        max_new_tokens=args.max_new_tokens,
        use_flash_attention=True,
        num_frames=0,
    )
    model_loader = create_loader(model_config)
    print("Loading model...")
    model_loader.load()
    print("Model loaded.\n")

    videos = sorted(grouped.keys())
    total_videos = len(videos)

    try:
        for video_idx, video_name in enumerate(videos, start=1):
            if video_name in completed_videos:
                continue

            print(f"[{video_idx}/{total_videos}] {video_name}")
            sys.stdout.flush()

            question_dicts = grouped[video_name]
            video_results = []
            video_correct = 0
            video_letter_parsed = 0

            for q_idx, q in enumerate(question_dicts, start=1):
                formatted_prompt = format_prompt(q)
                answers = q.get("answers") or q.get("all_answers") or []
                correct_index = q.get("correct_index", -1)

                try:
                    response = model_loader.generate_response(None, formatted_prompt)
                except Exception as e:
                    print(f"  ERROR on question {q_idx}: {e}", file=sys.stderr)
                    video_results.append(EvaluationResult(
                        video_name=video_name,
                        question_type=q.get("question_type", "unknown"),
                        is_trick=bool(q.get("is_trick", False)),
                        prompt=q.get("prompt", ""),
                        answers=answers,
                        correct_answer=q.get("correct_answer", ""),
                        correct_index=correct_index,
                        model_response="",
                        model_selected_index=None,
                        is_correct=False,
                        error=str(e),
                    ))
                    continue

                selected_index, is_correct, letter_parsed = score_response(
                    response, answers, correct_index
                )
                if is_correct:
                    video_correct += 1
                if letter_parsed:
                    video_letter_parsed += 1

                marker = "+" if is_correct else " "
                print(
                    f"  {marker} Q{q_idx} [{q.get('question_type','?')}]: "
                    f"{response!r} -> idx={selected_index}"
                )

                video_results.append(EvaluationResult(
                    video_name=video_name,
                    question_type=q.get("question_type", "unknown"),
                    is_trick=bool(q.get("is_trick", False)),
                    prompt=q.get("prompt", ""),
                    answers=answers,
                    correct_answer=q.get("correct_answer", ""),
                    correct_index=correct_index,
                    model_response=response,
                    model_selected_index=selected_index,
                    is_correct=is_correct,
                ))

            if video_results:
                checkpoint_entry = {
                    "video_name": video_name,
                    "timestamp": datetime.now().isoformat(),
                    "num_questions": len(video_results),
                    "num_correct": video_correct,
                    "num_letter_parsed": video_letter_parsed,
                    "results": [asdict(r) for r in video_results],
                }
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(checkpoint_entry, ensure_ascii=False) + "\n")

                acc = video_correct / len(video_results) * 100
                lp = video_letter_parsed / len(video_results) * 100
                print(f"  -> {video_correct}/{len(video_results)} ({acc:.1f}%)  letter_parsed: {lp:.1f}%\n")
                sys.stdout.flush()

    finally:
        print("Unloading model...")
        model_loader.unload()

    # Read all results from checkpoint (covers resumed runs too)
    all_results: list[EvaluationResult] = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                for r in json.loads(line)["results"]:
                    all_results.append(EvaluationResult(**r))

    summary = compute_summary(all_results, model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"evaluation_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"Total questions : {summary['total_questions']}")
    print(f"Correct         : {summary['correct_count']}")
    print(f"Accuracy        : {summary['accuracy']:.4f} ({summary['accuracy'] * 100:.2f}%)")
    print(f"Letter parsed   : {summary['letter_parsed_rate']*100:.1f}% (target >=95%; <90% means the model is not emitting letters)")
    print("\nAccuracy by type:")
    for qtype, stats in sorted(summary["accuracy_by_type"].items()):
        print(f"  {qtype:40s}: {stats['accuracy']*100:5.1f}%  ({stats['correct']:4d}/{stats['total']:4d})")
    print("\nAccuracy by trick:")
    for key, stats in sorted(summary["accuracy_by_trick"].items()):
        print(f"  {key:10s}: {stats['accuracy']*100:5.1f}%  ({stats['correct']:4d}/{stats['total']:4d})")


if __name__ == "__main__":
    main()
