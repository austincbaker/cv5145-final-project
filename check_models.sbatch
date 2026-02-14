#!/bin/bash
#SBATCH --job-name=check_models
#SBATCH --account=yrawat
#SBATCH --qos=yrawat
#SBATCH --time=2:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --output=eval_multi_%j.out
#SBATCH --error=eval_multi_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Model Cache Checker and Loader
# ==============================================================================
# This script checks which VLM models are already cached and downloads any
# missing ones from the HuggingFace Hub.
#
# Usage modes:
#   1. Check only (no download): sbatch check_models.sbatch
#   2. Check and download: sbatch check_models.sbatch --download
# ==============================================================================

# Print job info
echo "=============================================="
echo "VLM Model Cache Checker"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="
echo ""

# Configuration
DOWNLOAD_MODE="${1:-check}"  # "check" or "--download"

# Set HuggingFace cache location (optional - uses default if not set)
# export HF_HOME=/path/to/custom/cache

# Print cache location
if [ -n "$HF_HOME" ]; then
    echo "Using custom HuggingFace cache: $HF_HOME"
elif [ -n "$HUGGINGFACE_HUB_CACHE" ]; then
    echo "Using HuggingFace cache: $HUGGINGFACE_HUB_CACHE"
else
    echo "Using default HuggingFace cache: ~/.cache/huggingface"
fi
echo ""

# Ensure huggingface_hub is installed
echo "Checking dependencies..."
pip show huggingface_hub > /dev/null 2>&1 || pip install huggingface_hub
echo ""

# Run the checker
if [ "$DOWNLOAD_MODE" = "--download" ]; then
    echo "Mode: CHECK AND DOWNLOAD"
    echo "This will download any missing models from HuggingFace Hub"
    echo ""
    python check_and_load_models.py --download
else
    echo "Mode: CHECK ONLY"
    echo "This will only check which models are cached (no downloads)"
    echo "To download missing models, run: sbatch check_models.sbatch --download"
    echo ""
    python check_and_load_models.py
fi

EXIT_CODE=$?

# Print completion info
echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
