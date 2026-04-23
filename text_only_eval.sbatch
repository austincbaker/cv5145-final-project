#!/bin/bash
#SBATCH --job-name=text_only_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --output=text_only_eval_%j.out
#SBATCH --error=text_only_eval_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Text-Only Evaluation (No Video Frames)
# ==============================================================================
# Usage:
#   sbatch --export=MODEL=internvl2.5-8b text_only_eval.sbatch
#   sbatch --export=MODEL=llava-video-7b text_only_eval.sbatch
#   sbatch --export=MODEL=ovis2.5-9b text_only_eval.sbatch
#   sbatch --export=MODEL=videollama-7b text_only_eval.sbatch
#
# Run all 4:
#   for m in internvl2.5-8b llava-video-7b ovis2.5-9b videollama-7b; do
#     sbatch --export=MODEL=$m text_only_eval.sbatch
#   done
#
# Override questions file:
#   sbatch --export=MODEL=internvl2.5-8b,QUESTIONS_FILE=my_questions.json text_only_eval.sbatch
# ==============================================================================

# Environment setup
# Source conda init explicitly — .bashrc exits early for non-interactive shells
# (sbatch jobs), so the conda initialize block never runs.
eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh

module use ~/privatemodules
module load anaconda/25.11.1
conda activate base
module load cuda/12.6

# Configuration
ANNOTATIONS="annotations.json"
VIDEO_DIR="videos/"
MODEL="${MODEL:-qwen-7b}"
QUESTIONS_FILE="${QUESTIONS_FILE:-generated_questions_debiased_33pct.json}"
NUM_FRAMES="${NUM_FRAMES:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
STAGGER_DELAY=30

# Per-run output directory: model tag + job ID so multiple runs don't collide,
# and the directory stays stable across requeues of the same job.
MODEL_TAG="${MODEL//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_${MODEL_TAG}-${SLURM_JOB_ID}}_text_only"

# Track if we received a timeout signal
TIMEOUT_SIGNAL_RECEIVED=0

# Signal handler for graceful shutdown before timeout
handle_timeout_signal() {
    echo ""
    echo "=============================================="
    echo "TIMEOUT WARNING: Received USR1 signal"
    echo "Job will be requeued and resume from checkpoint"
    echo "Terminating workers gracefully..."
    echo "Time: $(date)"
    echo "=============================================="
    TIMEOUT_SIGNAL_RECEIVED=1

    pkill -P $$
    sleep 5

    echo "Requeueing job $SLURM_JOB_ID. Progress saved in checkpoints."
    scontrol requeue "$SLURM_JOB_ID"
}

trap 'handle_timeout_signal' USR1

# Print job info
echo "=============================================="
echo "Multi-GPU Evaluation (Subprocess Mode)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Questions file: ${QUESTIONS_FILE:-<on-the-fly generation>}"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Stagger delay: ${STAGGER_DELAY}s"
echo "Start time: $(date)"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "Auto-requeue: ENABLED (resumes from checkpoint on timeout)"
echo "=============================================="
echo ""

# Print GPU info
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Unset so each subprocess can set its own
unset CUDA_VISIBLE_DEVICES

echo "Starting parallel evaluation across $NUM_GPUS GPUs..."
echo "Each GPU will load its own model instance."
echo "Progress logs available in: $OUTPUT_DIR/logs/"
echo ""

# Each model family requires a specific transformers version.
if [[ "$MODEL" == *ovis* ]]; then
    REQUIRED_TF_VERSION="4.57.1"
elif [[ "$MODEL" == *internvl2* || "$MODEL" == *llava* ]]; then
    REQUIRED_TF_VERSION="4.49.0"
elif [[ "$MODEL" == *videollama* ]]; then
    REQUIRED_TF_VERSION="4.49.0"
elif [[ "$MODEL" == *Internvl3* ]]; then
    REQUIRED_TF_VERSION="4.52.1"
else
    REQUIRED_TF_VERSION="4.49.0"
fi

CURRENT_TF_VERSION=$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo "BROKEN")
echo "Current transformers version: $CURRENT_TF_VERSION"
echo "Required transformers version: $REQUIRED_TF_VERSION"

if [[ "$CURRENT_TF_VERSION" != "$REQUIRED_TF_VERSION" ]]; then
    echo "Switching transformers to $REQUIRED_TF_VERSION..."
    pip install transformers==$REQUIRED_TF_VERSION
    echo "Installed transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
else
    echo "Correct transformers version already installed"
fi


# Create output dir early so we can move the questions file into it before
# any workers start. This keeps all run artefacts together and is idempotent:
# on requeue the file is already in place, so we just update the path.
mkdir -p "$OUTPUT_DIR"

if [[ -n "$QUESTIONS_FILE" ]]; then
    QUESTIONS_BASENAME=$(basename "$QUESTIONS_FILE")
    QUESTIONS_DEST="${OUTPUT_DIR}/${QUESTIONS_BASENAME}"
    if [[ ! -f "$QUESTIONS_DEST" ]]; then
        cp "$QUESTIONS_FILE" "$QUESTIONS_DEST"
        echo "Moved questions file to: $QUESTIONS_DEST"
    else
        echo "Questions file already in output dir (resuming): $QUESTIONS_DEST"
    fi
    QUESTIONS_FILE="$QUESTIONS_DEST"
fi

# Build optional arguments
EXTRA_ARGS=""
if [[ -n "$QUESTIONS_FILE" ]]; then
    EXTRA_ARGS="--questions-file $QUESTIONS_FILE"
fi

# Run in background so we can catch signals
python text_only_eval.py \
    --questions-file "$QUESTIONS_FILE" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR" &

PYTHON_PID=$!

wait $PYTHON_PID
EXIT_CODE=$?

# Print completion info
echo ""
echo "=============================================="
if [ $TIMEOUT_SIGNAL_RECEIVED -eq 1 ]; then
    echo "Job interrupted for requeue: $(date)"
    echo "Will resume from checkpoint on next run"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully: $(date)"
else
    echo "Job failed with exit code $EXIT_CODE: $(date)"
fi
echo "=============================================="

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
