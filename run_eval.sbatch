#!/bin/bash
#SBATCH --job-name=eval_multi_gpu
#SBATCH --time=4:00:00
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --constraint=h100
#SBATCH --output=eval_multi_%j.out
#SBATCH --error=eval_multi_%j.err

# ==============================================================================
# Multi-GPU Parallel Evaluation Script (Subprocess Mode)
# ==============================================================================
# This script distributes videos across multiple GPUs using subprocess isolation.
# Each GPU worker is a completely separate Python process with its own CUDA context.
# 
# Key features:
# - Complete process isolation (avoids CUDA multiprocessing issues)
# - Staggered startup (30s default) to prevent resource contention
# - Each worker writes to its own checkpoint file
# - Automatic result merging at completion
# 
# Estimated speedup: Near-linear with GPU count
# 4 GPUs = ~4x faster than single GPU
# ==============================================================================

# Configuration
ANNOTATIONS="correct_annotations.json"
VIDEO_DIR="videos/"
OUTPUT_DIR="./results_multi"
MODEL="AIDC-AI/Ovis2.5-9B"
NUM_FRAMES=8
NUM_GPUS=4
STAGGER_DELAY=30  # Seconds between starting each GPU worker

# Print job info
echo "=============================================="
echo "Multi-GPU Evaluation (Subprocess Mode)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $NUM_GPUS"
echo "Stagger delay: ${STAGGER_DELAY}s"
echo "Start time: $(date)"
echo "=============================================="
echo ""

# Print GPU info
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# CRITICAL: Unset CUDA_VISIBLE_DEVICES so each subprocess can set its own
# If this is set, all subprocesses will try to use the same GPU
unset CUDA_VISIBLE_DEVICES

# Run parallel evaluation using subprocess-based runner
echo "Starting parallel evaluation across $NUM_GPUS GPUs..."
echo "Each GPU will load its own model instance."
echo "Progress logs available in: $OUTPUT_DIR/logs/"
echo ""

python -m prompt_generator.evaluation.parallel_runner \
    "$ANNOTATIONS" \
    "$VIDEO_DIR" \
    -o "$OUTPUT_DIR" \
    -g "$NUM_GPUS" \
    -m "$MODEL" \
    -f "$NUM_FRAMES" \
    --thinking-budget 512 \
    --max-new-tokens 1024 \
    --stagger-delay "$STAGGER_DELAY"

EXIT_CODE=$?

# Print completion info
echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully: $(date)"
else
    echo "Job failed with exit code $EXIT_CODE: $(date)"
fi
echo "=============================================="

# Print summary of output files
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""
echo "Checkpoint files:"
ls -la "$OUTPUT_DIR"/checkpoints/*.jsonl 2>/dev/null || echo "  (no checkpoint files found)"
echo ""
echo "Log files:"
ls -la "$OUTPUT_DIR"/logs/*.log 2>/dev/null || echo "  (no log files found)"

exit $EXIT_CODE
