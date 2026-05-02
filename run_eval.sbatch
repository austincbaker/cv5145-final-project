#!/bin/bash
#SBATCH --job-name=eval_optimized
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=h100
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# ==============================================================================
# Optimized Evaluation Script
# ==============================================================================
# This script runs the evaluation with all optimizations enabled:
# - Frame caching (extracts once per video)
# - Async prefetching (loads next video while processing current)
# - Hardware-accelerated video decoding (decord with GPU)
# - torch.compile for faster inference
# - Flash attention if available
# - Reduced token generation (128 instead of 1024)
# - Reduced thinking budget (256 instead of 512)
# ==============================================================================

# Configuration
ANNOTATIONS="annotations.json"
VIDEO_DIR="videos/"
OUTPUT_DIR="./results"
MODEL="AIDC-AI/Ovis2.5-9B"
NUM_FRAMES=8

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="
echo ""

# Print GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Install optional dependencies if not present
echo "Checking optional dependencies..."
pip show decord > /dev/null 2>&1 || echo "Note: decord not installed. Using OpenCV for video decoding."
pip show flash-attn > /dev/null 2>&1 || echo "Note: flash-attn not installed. Using standard attention."
echo ""

# Run evaluation with all optimizations
echo "Starting optimized evaluation..."
echo ""

python -m prompt_generator.evaluation.run_evaluation \
    "$ANNOTATIONS" \
    "$VIDEO_DIR" \
    --all-questions \
    -o "$OUTPUT_DIR" \
    -m "$MODEL" \
    -f "$NUM_FRAMES" \
    --thinking-budget 256 \
    --max-new-tokens 128 \
    --prefetch-count 2

# Print completion info
echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="
