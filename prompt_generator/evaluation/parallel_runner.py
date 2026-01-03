"""
Multi-GPU parallel evaluation runner using subprocess isolation.

Distributes videos across multiple GPUs using separate processes.
Each GPU worker is a completely independent subprocess with its own
CUDA context, avoiding the common issues with Python multiprocessing and CUDA.
"""
import json
import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse
from collections import defaultdict


def merge_checkpoints(checkpoint_dir: Path, output_path: Path, config: dict) -> dict:
    """Merge checkpoint files from all GPUs into final output."""
    all_results = []
    video_stats = {}
    
    # Read all checkpoint files
    for checkpoint_file in sorted(checkpoint_dir.glob("checkpoint_gpu*.jsonl")):
        print(f"Reading {checkpoint_file.name}...")
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    video_name = data["video_name"]
                    video_stats[video_name] = {
                        "gpu_id": data["gpu_id"],
                        "num_questions": data["num_questions"],
                        "num_correct": data.get("num_correct", 0),
                        "timestamp": data["timestamp"],
                    }
                    all_results.extend(data["results"])
    
    # Calculate summary statistics
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct", False))
    accuracy = correct / total if total > 0 else 0.0
    
    # Accuracy by question type
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
            "accuracy": c / t if t > 0 else 0.0,
        }
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": config.get("model_path", "unknown"),
        "num_frames": config.get("num_frames", 8),
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
    stagger_delay: float = 30.0,
) -> dict:
    """
    Run evaluation across multiple GPUs using subprocess isolation.
    
    Each GPU runs in a completely separate Python process with its own
    CUDA context. This avoids the common issues with multiprocessing.Pool
    and CUDA initialization.
    
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
        stagger_delay: Seconds to wait between starting each GPU worker
    
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
    
    print("=" * 70)
    print("MULTI-GPU PARALLEL EVALUATION (Subprocess Mode)")
    print("=" * 70)
    print(f"Total videos: {len(video_names)}")
    print(f"GPUs requested: {num_gpus}")
    print(f"Model: {model_path}")
    print(f"Stagger delay: {stagger_delay}s between GPU starts")
    print()
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    videos_dir = output_path / "video_assignments"
    videos_dir.mkdir(exist_ok=True)
    
    logs_dir = output_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Clear old files
    for old_file in checkpoint_dir.glob("checkpoint_gpu*.jsonl"):
        old_file.unlink()
    for old_file in videos_dir.glob("videos_gpu*.json"):
        old_file.unlink()
    
    # Split videos across GPUs (round-robin for better load balancing)
    video_batches = [[] for _ in range(num_gpus)]
    for i, video in enumerate(video_names):
        video_batches[i % num_gpus].append(video)
    
    print("Video distribution:")
    for i, batch in enumerate(video_batches):
        print(f"  GPU {i}: {len(batch)} videos")
    print()
    
    # Write video assignments to files
    video_files = []
    for gpu_id in range(num_gpus):
        video_file = videos_dir / f"videos_gpu{gpu_id}.json"
        with open(video_file, 'w') as f:
            json.dump(video_batches[gpu_id], f)
        video_files.append(video_file)
    
    # Get the path to the worker script
    worker_module = "prompt_generator.evaluation.gpu_worker"
    
    # Start worker processes with staggered delays
    processes = []
    log_files = []
    
    print(f"Starting {num_gpus} GPU worker processes...")
    print()
    
    for gpu_id in range(num_gpus):
        if not video_batches[gpu_id]:
            print(f"GPU {gpu_id}: No videos assigned, skipping")
            continue
        
        # Create log files for this worker
        stdout_log = logs_dir / f"gpu{gpu_id}_stdout.log"
        stderr_log = logs_dir / f"gpu{gpu_id}_stderr.log"
        
        stdout_file = open(stdout_log, 'w')
        stderr_file = open(stderr_log, 'w')
        log_files.extend([stdout_file, stderr_file])
        
        # Build command
        cmd = [
            sys.executable, "-m", worker_module,
            "--gpu-id", str(gpu_id),
            "--videos-file", str(video_files[gpu_id]),
            "--annotations", annotations_path,
            "--video-dir", video_dir,
            "--checkpoint-dir", str(checkpoint_dir),
            "--model-path", model_path,
            "--num-frames", str(num_frames),
            "--num-distractors", str(num_distractors),
            "--thinking-budget", str(thinking_budget),
            "--max-new-tokens", str(max_new_tokens),
        ]
        
        # Set environment with CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        print(f"Starting GPU {gpu_id} worker (CUDA_VISIBLE_DEVICES={gpu_id})...")
        
        # Start the process
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
            cwd=os.getcwd(),
        )
        processes.append((gpu_id, proc))
        
        # Stagger startup to avoid resource contention during model loading
        if gpu_id < num_gpus - 1 and stagger_delay > 0:
            print(f"  Waiting {stagger_delay}s before starting next worker...")
            time.sleep(stagger_delay)
    
    print()
    print(f"All {len(processes)} workers started. Monitoring progress...")
    print(f"Logs available in: {logs_dir}")
    print()
    
    # Monitor processes
    start_time = time.time()
    completed = set()
    failed = []
    
    while len(completed) < len(processes):
        time.sleep(10)  # Check every 10 seconds
        
        for gpu_id, proc in processes:
            if gpu_id in completed:
                continue
            
            ret = proc.poll()
            if ret is not None:
                completed.add(gpu_id)
                elapsed = time.time() - start_time
                
                if ret == 0:
                    # Read last lines of log for summary
                    stdout_log = logs_dir / f"gpu{gpu_id}_stdout.log"
                    try:
                        with open(stdout_log, 'r') as f:
                            lines = f.readlines()
                            # Find the FINISHED line
                            for line in reversed(lines):
                                if "FINISHED" in line:
                                    print(f"GPU {gpu_id} completed: {line.strip()}")
                                    break
                            else:
                                print(f"GPU {gpu_id} completed (exit code 0) after {elapsed:.1f}s")
                    except:
                        print(f"GPU {gpu_id} completed (exit code 0) after {elapsed:.1f}s")
                else:
                    failed.append(gpu_id)
                    print(f"GPU {gpu_id} FAILED with exit code {ret} after {elapsed:.1f}s")
                    # Print last few lines of stderr
                    stderr_log = logs_dir / f"gpu{gpu_id}_stderr.log"
                    try:
                        with open(stderr_log, 'r') as f:
                            lines = f.readlines()[-10:]
                            print(f"  Last error lines:")
                            for line in lines:
                                print(f"    {line.rstrip()}")
                    except:
                        pass
        
        # Print progress summary
        active = len(processes) - len(completed)
        if active > 0:
            elapsed = time.time() - start_time
            print(f"  [{elapsed/60:.1f}m elapsed] {len(completed)}/{len(processes)} workers completed, {active} active")
    
    # Close log files
    for f in log_files:
        f.close()
    
    total_time = time.time() - start_time
    print()
    print(f"All workers finished in {total_time/60:.1f} minutes")
    
    if failed:
        print(f"WARNING: {len(failed)} workers failed: GPU(s) {failed}")
        print(f"Check logs in {logs_dir} for details")
    
    # Merge results from checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = output_path / f"evaluation_{timestamp}.json"
    
    print()
    print("Merging results from checkpoints...")
    
    final_results = merge_checkpoints(
        checkpoint_dir, 
        final_output, 
        {"model_path": model_path, "num_frames": num_frames}
    )
    
    # Print final summary
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total videos evaluated: {final_results['total_videos_evaluated']}")
    print(f"Total questions: {final_results['total_questions']}")
    print(f"Correct: {final_results['correct_count']}")
    print(f"Overall accuracy: {final_results['accuracy']:.2%}")
    print()
    print("Accuracy by question type:")
    for qtype, stats in sorted(final_results['accuracy_by_type'].items()):
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
    print()
    print(f"Results saved to: {final_output}")
    print(f"Checkpoints in: {checkpoint_dir}")
    print(f"Logs in: {logs_dir}")
    
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
    parser.add_argument(
        "--stagger-delay",
        type=float,
        default=30.0,
        help="Seconds to wait between starting each GPU worker (default: 30)",
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
        stagger_delay=args.stagger_delay,
    )


if __name__ == "__main__":
    main()