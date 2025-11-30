import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from typing import Optional

from ..generator import QuestionGenerator, GeneratedQuestion
from .model_loader import OvisModelLoader, ModelConfig
from .video_processor import VideoProcessor, VideoProcessorConfig


@dataclass
class VideoCheckpoint:
    """Checkpoint data for a single video."""
    video_name: str
    timestamp: str
    num_questions: int
    results: list[dict]
    completed: bool = True


@dataclass
class EvaluationResult:
    video_name: str
    question_type: str
    prompt: str
    answers: list[str]
    correct_answer: str
    correct_index: int
    model_response: str
    model_selected_index: int | None
    is_correct: bool
    error: str | None = None


@dataclass
class EvaluationSummary:
    total_questions: int
    correct_count: int
    accuracy: float
    accuracy_by_type: dict[str, dict]
    results: list[EvaluationResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_path: str = ""
    num_frames: int = 8


class VideoQuestionEvaluator:
    def __init__(
        self,
        annotations: list[dict],
        video_dir: str | Path,
        model_config: ModelConfig | None = None,
        video_config: VideoProcessorConfig | None = None,
        num_distractors: int = 5,
    ):
        self.video_dir = Path(video_dir)
        self.model_config = model_config or ModelConfig()
        self.video_config = video_config or VideoProcessorConfig()
        self.num_distractors = num_distractors
        
        # Filter annotations so they only include videos that exist
        self.available_videos = self._scan_available_videos()
        self.annotations, flattened_annotations = self._filter_annotations_by_videos(annotations)
        
        if not self.annotations:
            # Show mismatch diagnostics
            print("\n" + "=" * 70, file=sys.stderr)
            print("ERROR: No matching videos found", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            
            # Show sample video names from directory
            sample_videos = sorted(list(self.available_videos))[:5]
            print(f"\nSample video filenames in directory ({len(self.available_videos)} total):", file=sys.stderr)
            for v in sample_videos:
                print(f"  - {v}", file=sys.stderr)
            
            # Show sample video names from annotations
            sample_annotations = []
            for entry in flattened_annotations[:5]:
                if isinstance(entry, dict):
                    sample_annotations.append(entry.get("video_name", "MISSING"))
            
            print(f"\nSample video names in annotations ({len(flattened_annotations)} total):", file=sys.stderr)
            for v in sample_annotations:
                print(f"  - {v}", file=sys.stderr)
            
            # Try to find matches with suggestions
            print("\nTrying to find matches with fuzzy matching:", file=sys.stderr)
            suggestions_found = 0
            for annotation_name in sample_annotations[:5]:
                suggestion = self._suggest_video_name_fix(annotation_name)
                if suggestion and suggestion != annotation_name:
                    print(f"  '{annotation_name}' → '{suggestion}' (found!)", file=sys.stderr)
                    suggestions_found += 1
                else:
                    print(f"  '{annotation_name}' → No match found", file=sys.stderr)
            
            if suggestions_found > 0:
                print(f"\nFound {suggestions_found} potential matches! ", file=sys.stderr)
                print("Your annotations likely need extension fixes (.mp4, etc.)", file=sys.stderr)
            
            print("\nPossible issues:", file=sys.stderr)
            print("  1. File extension mismatch (.mp4 vs .avi vs no extension)", file=sys.stderr)
            print("  2. Different naming convention (underscores vs hyphens)", file=sys.stderr)
            print("  3. Wrong video directory path", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            
            raise ValueError(
                f"No videos found in {video_dir} matching annotation entries. "
                f"Found {len(self.available_videos)} videos in directory but none match annotations. "
                f"See diagnostic output above for details."
            )
        
        print(f"Found {len(self.available_videos)} videos in directory")
        print(f"Matched {len(self.annotations)} annotations with available videos")
        
        self.question_generator = QuestionGenerator(
            self.annotations, num_distractors=num_distractors
        )
        self.model_loader = OvisModelLoader(self.model_config)
        self.video_processor = VideoProcessor(self.video_config)

    def _scan_available_videos(self) -> set[str]:
        """Scan video directory and return set of available video filenames."""
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        available = set()
        
        for file in self.video_dir.iterdir():
            if file.is_file() and file.suffix.lower() in video_extensions:
                available.add(file.name)
        
        return available
    
    def _suggest_video_name_fix(self, annotation_name: str) -> str | None:
        """Try to find a matching video by adding/removing extensions."""
        if annotation_name in self.available_videos:
            return annotation_name
        
        common_exts = [".mp4", ".avi", ".mov", ".mkv"]
        for ext in common_exts:
            candidate = annotation_name + ext
            if candidate in self.available_videos:
                return candidate
        
        if "." in annotation_name:
            base_name = annotation_name.rsplit(".", 1)[0]
            for ext in common_exts:
                candidate = base_name + ext
                if candidate in self.available_videos:
                    return candidate
        
        return None

    def _filter_annotations_by_videos(self, annotations: list[dict]) -> tuple[list[dict], list[dict]]:
        """Filter annotations to only include entries with available videos.
        
        Returns: tuple: (filtered_annotations, flattened_annotations)
        """
        # Handle nested list structure
        flattened_annotations = []
        for entry in annotations:
            if isinstance(entry, list):
                # Nested list needs to be flattened
                flattened_annotations.extend(entry)
            elif isinstance(entry, dict):
                # Already a dict
                flattened_annotations.append(entry)
            else:
                print(f"Warning: Unexpected annotation type: {type(entry)}", file=sys.stderr)
                continue
        
        # Filter by available videos
        filtered = []
        skipped = []
        
        for entry in flattened_annotations:
            if not isinstance(entry, dict):
                print(f"Warning: Skipping non-dict entry: {type(entry)}", file=sys.stderr)
                continue
                
            video_name = entry.get("video_name", "")
            if video_name in self.available_videos:
                filtered.append(entry)
            else:
                skipped.append(video_name)
        
        if skipped:
            print(f"Skipped {len(skipped)} annotations with missing videos")
            if len(skipped) <= 10:
                print(f"  Missing: {', '.join(skipped)}")
        
        return filtered, flattened_annotations

    def get_availability_stats(self) -> dict:
        """Get statistics about video availability."""
        return {
            "total_videos_in_dir": len(self.available_videos),
            "total_annotations": len(self.annotations),
            "matched": len(self.annotations),
            "available_videos": sorted(list(self.available_videos)),
        }
    
    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, VideoCheckpoint]:
        """Load checkpoint file and return completed videos."""
        completed = {}
        
        if not checkpoint_path.exists():
            return completed
        
        try:
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        checkpoint = VideoCheckpoint(**data)
                        if checkpoint.completed:
                            completed[checkpoint.video_name] = checkpoint
        except Exception as e:
            print(f"Warning: Error loading checkpoint: {e}", file=sys.stderr)
            print("Starting fresh evaluation", file=sys.stderr)
            return {}
        
        return completed
    
    def save_video_checkpoint(
        self, 
        checkpoint_path: Path, 
        video_name: str, 
        results: list[EvaluationResult]
    ) -> None:
        # Append a video's results to the checkpoint file.
        checkpoint = VideoCheckpoint(
            video_name=video_name,
            timestamp=datetime.now().isoformat(),
            num_questions=len(results),
            results=[asdict(r) for r in results],
            completed=True
        )
        
        # Append to checkpoint file (JSONL format)
        with open(checkpoint_path, 'a') as f:
            f.write(json.dumps(asdict(checkpoint)) + '\n')
    
    def convert_checkpoint_to_final(
        self, 
        checkpoint_path: Path, 
        output_path: Path
    ) -> dict:
        # Convert checkpoint file JSON output.
        all_results = []
        video_stats = {}
        
        # Read checkpoints
        with open(checkpoint_path, 'r') as f:
            for line in f:
                if line.strip():
                    checkpoint = VideoCheckpoint(**json.loads(line))
                    # Convert dict to EvaluationResult 
                    for result_dict in checkpoint.results:
                        all_results.append(EvaluationResult(**result_dict))
                    video_stats[checkpoint.video_name] = {
                        'num_questions': checkpoint.num_questions,
                        'timestamp': checkpoint.timestamp
                    }
        
        summary = self._compute_summary(all_results)
        
        output_data = {
            "timestamp": summary.timestamp,
            "model_path": summary.model_path,
            "num_frames": summary.num_frames,
            "total_questions": summary.total_questions,
            "correct_count": summary.correct_count,
            "accuracy": summary.accuracy,
            "accuracy_by_type": summary.accuracy_by_type,
            "total_videos_evaluated": len(video_stats),
            "video_stats": video_stats,
            "results": [asdict(r) for r in summary.results],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_data

    def load_model(self) -> None:
        self.model_loader.load()

    def unload_model(self) -> None:
        self.model_loader.unload()

    def evaluate_question(
        self, question: GeneratedQuestion
    ) -> EvaluationResult:
        video_path = self.video_dir / question.video_name

        try:
            frames = self.video_processor.extract_frames(video_path)
        except (FileNotFoundError, RuntimeError) as e:
            return EvaluationResult(
                video_name=question.video_name,
                question_type=question.question_type,
                prompt=question.prompt,
                answers=question.answers,
                correct_answer=question.correct_answer,
                correct_index=question.correct_index,
                model_response="",
                model_selected_index=None,
                is_correct=False,
                error=str(e),
            )

        formatted_prompt = self._format_prompt(question)

        try:
            response = self.model_loader.generate_response(frames, formatted_prompt)
        except Exception as e:
            return EvaluationResult(
                video_name=question.video_name,
                question_type=question.question_type,
                prompt=question.prompt,
                answers=question.answers,
                correct_answer=question.correct_answer,
                correct_index=question.correct_index,
                model_response="",
                model_selected_index=None,
                is_correct=False,
                error=f"Model error: {str(e)}",
            )

        selected_index = self._parse_model_response(response, question.answers)
        is_correct = selected_index == question.correct_index

        return EvaluationResult(
            video_name=question.video_name,
            question_type=question.question_type,
            prompt=question.prompt,
            answers=question.answers,
            correct_answer=question.correct_answer,
            correct_index=question.correct_index,
            model_response=response,
            model_selected_index=selected_index,
            is_correct=is_correct,
        )

    def evaluate_batch(
        self,
        questions: list[GeneratedQuestion],
        progress_callback=None,
    ) -> EvaluationSummary:
        results = []

        for i, question in enumerate(questions):
            result = self.evaluate_question(question)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(questions), result)

        return self._compute_summary(results)

    def evaluate_random(
        self,
        num_questions: int,
        progress_callback=None,
    ) -> EvaluationSummary:
        questions = self.question_generator.generate_questions(
            count=num_questions, allow_duplicates=False
        )
        return self.evaluate_batch(questions, progress_callback)
    
    def evaluate_all_with_checkpoint(
        self,
        checkpoint_path: Path,
        output_path: Path,
        resume: bool = True,
        max_retries: int = 3,
        progress_callback=None,
    ) -> dict:
        """
        Evaluate all questions for all videos with checkpointing.
        """
        # Load checkpoint if resuming
        completed_videos = {}
        if resume:
            completed_videos = self.load_checkpoint(checkpoint_path)
            if completed_videos:
                print(f"Found checkpoint with {len(completed_videos)} completed videos")
                print(f"Resuming evaluation...")
        
        all_videos = {}
        for annotation in self.annotations:
            video_name = annotation.get('video_name')
            if video_name:
                if video_name not in all_videos:
                    all_videos[video_name] = []
                all_videos[video_name].append(annotation)
        
        if resume and completed_videos:
            remaining_videos = {
                v: anns for v, anns in all_videos.items()
                if v not in completed_videos
            }
        else:
            remaining_videos = all_videos
        
        total_videos = len(all_videos)
        completed_count = len(completed_videos)
        remaining_count = len(remaining_videos)
        
        print(f"\nTotal videos: {total_videos}")
        print(f"Completed: {completed_count}")
        print(f"Remaining: {remaining_count}")
        print()
        
        # Process each video
        failed_videos = []
        
        for video_idx, (video_name, video_annotations) in enumerate(sorted(remaining_videos.items()), start=1):
            current_video_num = completed_count + video_idx
            
            if progress_callback:
                progress_callback(current_video_num, total_videos, None, 
                                video_name=video_name, status="starting")
            
            # Create temporary generator for just this video's annotations
            from ..generator import QuestionGenerator
            temp_generator = QuestionGenerator(video_annotations, self.num_distractors)
            video_questions = temp_generator.generate_all_questions()
            
            if not video_questions:
                print(f"  Warning: No questions generated for {video_name}", file=sys.stderr)
                continue
            
            retry_count = 0
            success = False
            video_results = []
            last_error = None
            
            while retry_count <= max_retries and not success:
                try:
                    video_results = []
                    for q_idx, question in enumerate(video_questions, start=1):
                        result = self.evaluate_question(question)
                        video_results.append(result)
                        
                        if progress_callback:
                            progress_callback(
                                current_video_num, total_videos, result,
                                video_name=video_name,
                                question_num=q_idx,
                                total_questions=len(video_questions)
                            )
                    
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    last_error = str(e)
                    if retry_count <= max_retries:
                        print(f"  Error on {video_name} (attempt {retry_count}/{max_retries}): {e}")
                        print(f"  Retrying...")
                    else:
                        print(f"  ✗ Failed {video_name} after {max_retries} retries: {e}")
            
            if success:
                # Save checkpoint
                self.save_video_checkpoint(checkpoint_path, video_name, video_results)
                
                if progress_callback:
                    progress_callback(current_video_num, total_videos, None,
                                    video_name=video_name, status="completed")
            else:
                failed_videos.append((video_name, last_error))
                print(f"  Skipping {video_name} after failed retries")
        
        print("\nConverting checkpoint to final JSON...")
        final_results = self.convert_checkpoint_to_final(checkpoint_path, output_path)
        
        if failed_videos:
            print(f"\n{len(failed_videos)} videos failed after retries:")
            for video_name, error in failed_videos:
                print(f"  - {video_name}: {error}")
        
        print(f"\n✓ Evaluation complete!")
        print(f"✓ Results saved to {output_path}")
        print(f"✓ Checkpoint saved to {checkpoint_path}")
        
        return final_results

    def _format_prompt(self, question: GeneratedQuestion) -> str:
        lines = [
            "Watch this video carefully and answer the following multiple-choice question.",
            "Select ONLY the number (1, 2, 3, etc.) of the correct answer.",
            "",
            f"Question: {question.prompt}",
            "",
            "Options:",
        ]

        for i, answer in enumerate(question.answers):
            lines.append(f"{i + 1}. {answer}")

        lines.extend([
            "",
            "Answer with ONLY the option number (e.g., '1' or '2').",
        ])

        return "\n".join(lines)

    def _parse_model_response(
        self, response: str, answers: list[str]
    ) -> int | None:
        response_clean = response.strip()

        single_digit = re.search(r"^(\d)$", response_clean)
        if single_digit:
            idx = int(single_digit.group(1)) - 1
            if 0 <= idx < len(answers):
                return idx

        patterns = [
            r"(?:answer|option|choice)[\s:]*(\d)",
            r"^(\d)\.",
            r"^\((\d)\)",
            r"(\d)(?:\s|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(answers):
                    return idx

        response_lower = response_clean.lower()
        for i, answer in enumerate(answers):
            if answer.lower() in response_lower:
                return i

        return None

    def _compute_summary(self, results: list[EvaluationResult]) -> EvaluationSummary:
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0

        by_type = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in results:
            by_type[r.question_type]["total"] += 1
            if r.is_correct:
                by_type[r.question_type]["correct"] += 1

        accuracy_by_type = {}
        for qtype, counts in by_type.items():
            t, c = counts["total"], counts["correct"]
            accuracy_by_type[qtype] = {
                "total": t,
                "correct": c,
                "accuracy": c / t if t > 0 else 0.0,
            }

        return EvaluationSummary(
            total_questions=total,
            correct_count=correct,
            accuracy=accuracy,
            accuracy_by_type=accuracy_by_type,
            results=results,
            model_path=self.model_config.model_path,
            num_frames=self.video_config.num_frames,
        )


def save_evaluation_results(
    summary: EvaluationSummary, output_dir: str | Path
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{timestamp}.json"
    filepath = output_path / filename

    output_data = {
        "timestamp": summary.timestamp,
        "model_path": summary.model_path,
        "num_frames": summary.num_frames,
        "total_questions": summary.total_questions,
        "correct_count": summary.correct_count,
        "accuracy": summary.accuracy,
        "accuracy_by_type": summary.accuracy_by_type,
        "results": [asdict(r) for r in summary.results],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return str(filepath)