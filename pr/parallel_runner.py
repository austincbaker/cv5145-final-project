"""
Multi-GPU parallel evaluation runner.

Distributes videos across multiple GPUs for faster evaluation.
"""
import json
import sys
import os
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from multiprocessing import Pool, Manager, set_start_method
from typing import Optional
import argparse


def worker_init(gpu_id: int):
    """Initialize worker process with specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    

def process_video_batch(args: tuple) -> list[dict]:
    """
    Process a batch of videos on a single GPU.
    
    Args:
        args: Tuple of (gpu_id, video_names, config_dict, annotations_path, video_dir)
    
    Returns:
        List of result dictionaries
    """
    gpu_id, video_names, config_dict, annotations_path, video_dir, checkpoint_dir = args
    
    # Set GPU before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Import after setting CUDA device
    import torch
    from .model_loader import ModelConfig, OvisModelLoader
    from .video_processor import VideoProcessorConfig, create_video_processor
    from .evaluator import VideoQuestionEvaluator, EvaluationResult
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    if isinstance(annotations, dict) and "annotations" in annotations:
        annotations = annotations["annotations"]
    
    # Filter to only videos in this batch
    batch_annotations = [
        a for a in annotations 
        if a.get("video_name") in video_names
    ]
    
    # Create configs
    model_config = ModelConfig(**config_dict["model"])
    video_config = VideoProcessorConfig(**config_dict["video"])
    
    # Create evaluator
    try:
        evaluator = VideoQuestionEvaluator(
            annotations=batch_annotations,
            video_dir=video_dir,
            model_config=model_config,
            video_config=video_config,
            num_distractors=config_dict.get("num_distractors", 5),
            use_async_loading=False,  # Disable async in parallel mode
        )
    except ValueError as e:
        print(f"GPU {gpu_id}: Error creating evaluator: {e}", file=sys.stderr)
        return []
    
    # Load model
    print(f"GPU {gpu_id}: Loading model...")
    evaluator.load_model()
    print(f"GPU {gpu_id}: Model loaded, processing {len(video_names)} videos")
    
    all_results = []
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_gpu{gpu_id}.jsonl"
    
    try:
        for idx, video_name in enumerate(video_names, start=1):
            print(f"GPU {gpu_id}: [{idx}/{len(video_names)}] {video_name}")
            
            # Get annotations for this video
            video_annotations = [
                a for a in batch_annotations 
                if a.get("video_name") == video_name
            ]
            
            if not video_annotations:
                continue
            
            # Generate questions
            from ..generator import QuestionGenerator
            temp_generator = QuestionGenerator(
                video_annotations, 
                config_dict.get("num_distractors", 5)
            )
            questions = temp_generator.generate_all_questions()
            
            if not questions:
                continue
            
            # Evaluate
            results = evaluator.evaluate_video_questions(
                video_name, 
                questions,
                video_num=idx,
                total_videos=len(video_names)
            )
            
            # Save checkpoint
            checkpoint_data = {
                "video_name": video_name,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat(),
                "results": [asdict(r) for r in results]
            }
            
            with open(checkpoint_path, 'a') as f:
                f.write(json.dumps(checkpoint_data) + '\n')
            
            all_results.extend([asdict(r) for r in results])
    
    finally:
        evaluator.unload_model()
    
    print(f"GPU {gpu_id}: Completed {len(video_names)} videos, {len(all_results)} questions")
    return all_results


def merge_checkpoints(checkpoint_dir: Path, output_path: Path, model_config: dict) -> dict:
    """Merge checkpoint files from all GPUs into final output."""
    from collections import defaultdict
    
    all_results = []
    video_stats = {}
    
    # Read all checkpoint files
    for checkpoint_file in checkpoint_dir.glob("checkpoint_gpu*.jsonl"):
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    video_name = data["video_name"]
                    results = data["results"]
                    
                    all_results.extend(results)
                    video_stats[video_name] = {
                        "num_questions": len(results),
                        "gpu_id": data.get("gpu_id"),
                        "timestamp": data["timestamp"]
                    }
    
    # Calculate statistics
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct", False))
    accuracy = correct / total if total > 0 else 0.0
    
    # Accuracy by type
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in all_results:
        qtype = r.get("question_type", "unknown")
        by_type[qtype]["total"] += 1
        if r.get("is_correct", False):
            by_type[qtype]["correct"] += 1
    
    accuracy_by_type = {}
    for qtype, counts in by_type.items():
        t, c = counts["total"], counts["correct"]
        accuracy_by_type[qtype] = {
            "total": t,
            "correct": c,
            "accuracy": c / t if t > 0 else 0.0
        }
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_config.get("model_path", ""),
        "num_frames": model_config.get("num_frames", 8),
        "total_questions": total,
        "correct_count": correct,
        "accuracy": accuracy,
        "accuracy_by_type": accuracy_by_type,
        "total_videos_evaluated": len(video_stats),
        "video_stats": video_stats,
        "results": all_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_data


def run_parallel_evaluation(
    annotations_path: str,
    video_dir: str,
    output_dir: str,
    num_gpus: int = 1,
    model_path: str = "AIDC-AI/Ovis2.5-9B",
    num_frames: int = 8,
    num_distractors: int = 5,
    thinking_budget: int = 256,
    max_new_tokens: int = 128,
) -> dict:
    """
    Run evaluation across multiple GPUs.
    
    Args:
        annotations_path: Path to annotations JSON
        video_dir: Directory containing videos
        output_dir: Directory for output files
        num_gpus: Number of GPUs to use
        model_path: HuggingFace model path
        num_frames: Number of frames per video
        num_distractors: Number of distractor answers
        thinking_budget: Model thinking budget
        max_new_tokens: Max tokens to generate
    
    Returns:
        Final evaluation results dictionary
    """
    # Load annotations to get video list
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    if isinstance(annotations, dict) and "annotations" in annotations:
        annotations = annotations["annotations"]
    
    # Get unique video names
    video_names = sorted(set(a.get("video_name") for a in annotations if a.get("video_name")))
    
    print(f"Total videos: {len(video_names)}")
    print(f"Using {num_gpus} GPU(s)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Split videos across GPUs
    video_batches = [video_names[i::num_gpus] for i in range(num_gpus)]
    
    for i, batch in enumerate(video_batches):
        print(f"GPU {i}: {len(batch)} videos")
    
    # Prepare config dict
    config_dict = {
        "model": {
            "model_path": model_path,
            "thinking_budget": thinking_budget,
            "max_new_tokens": max_new_tokens,
            "use_torch_compile": True,
            "use_flash_attention": True,
        },
        "video": {
            "num_frames": num_frames,
        },
        "num_distractors": num_distractors,
    }
    
    # Prepare arguments for each worker
    worker_args = [
        (
            gpu_id,
            video_batches[gpu_id],
            config_dict,
            annotations_path,
            video_dir,
            str(checkpoint_dir),
        )
        for gpu_id in range(num_gpus)
    ]
    
    # Run parallel evaluation
    if num_gpus == 1:
        # Single GPU - run directly
        all_results = process_video_batch(worker_args[0])
    else:
        # Multi-GPU - use multiprocessing
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        with Pool(num_gpus) as pool:
            results_lists = pool.map(process_video_batch, worker_args)
        
        all_results = []
        for results in results_lists:
            all_results.extend(results)
    
    # Merge results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = output_path / f"evaluation_{timestamp}.json"
    
    final_results = merge_checkpoints(
        checkpoint_dir, 
        final_output, 
        {"model_path": model_path, "num_frames": num_frames}
    )
    
    print(f"\nEvaluation complete!")
    print(f"Total questions: {final_results['total_questions']}")
    print(f"Accuracy: {final_results['accuracy']:.2%}")
    print(f"Results saved to: {final_output}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel evaluation across multiple GPUs"
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
        "-o", "--output-dir",
        default="./results",
        help="Directory for output files (default: ./results)",
    )
    parser.add_argument(
        "-g", "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "-m", "--model-path",
        default="AIDC-AI/Ovis2.5-9B",
        help="HuggingFace model path (default: AIDC-AI/Ovis2.5-9B)",
    )
    parser.add_argument(
        "-f", "--num-frames",
        type=int,
        default=8,
        help="Number of frames per video (default: 8)",
    )
    parser.add_argument(
        "-d", "--num-distractors",
        type=int,
        default=5,
        help="Number of distractor answers (default: 5)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=256,
        help="Model thinking budget (default: 256)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate (default: 128)",
    )
    
    args = parser.parse_args()
    
    run_parallel_evaluation(
        annotations_path=args.annotations_json,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        model_path=args.model_path,
        num_frames=args.num_frames,
        num_distractors=args.num_distractors,
        thinking_budget=args.thinking_budget,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()