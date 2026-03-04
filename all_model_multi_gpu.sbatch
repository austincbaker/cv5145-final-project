#!/bin/bash
#SBATCH --job-name=bullying_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --output=all_model_multi_gpu_%j.out
#SBATCH --error=all_model_multi_gpu_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Multi-GPU Parallel Evaluation Script
# ==============================================================================
# Supports model selection via:
#   sbatch --export=MODEL=qwen-72b run_eval_multi_gpu.sbatch
#   sbatch --export=MODEL=internvl-78b run_eval_multi_gpu.sbatch
#   sbatch --export=MODEL=llama-90b run_eval_multi_gpu.sbatch
#   sbatch --export=MODEL=nvila-15b run_eval_multi_gpu.sbatch
#
# Or with a full HuggingFace path:
#   sbatch --export=MODEL=Qwen/Qwen2.5-VL-72B-Instruct run_eval_multi_gpu.sbatch
#
# If MODEL is not set, defaults to ovis-9b.
# ==============================================================================

# Environment setup
module use ~/privatemodules
module load anaconda/25.11.1
conda activate base
module load cuda/12.6
# squashfuse videos.squashfs videos/ || { echo "ERROR: squashfuse failed to mount videos.squashfs"; exit 1; }
# if [ -z "$(ls -A videos/)" ]; then
#     echo "ERROR: videos/ mount is empty after squashfuse"
#     exit 1
# fi
# echo "Mounted videos.squashfs successfully ($(ls videos/ | wc -l) files)"

# Configuration
ANNOTATIONS="annotations.json"
VIDEO_DIR="videos/"
MODEL="${MODEL:-qwen-7b}"
QUESTIONS_FILE="${QUESTIONS_FILE:-}"
NUM_FRAMES=8
NUM_GPUS="${NUM_GPUS:-1}"
STAGGER_DELAY=30

# Per-run output directory: model tag + job ID so multiple runs don't collide,
# and the directory stays stable across requeues of the same job.
MODEL_TAG="${MODEL//\//_}"
OUTPUT_DIR="./results_${MODEL_TAG}-${SLURM_JOB_ID}"

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

# Create output dir early so we can move the questions file into it before
# any workers start. This keeps all run artefacts together and is idempotent:
# on requeue the file is already in place, so we just update the path.
mkdir -p "$OUTPUT_DIR"

if [[ -n "$QUESTIONS_FILE" ]]; then
    QUESTIONS_BASENAME=$(basename "$QUESTIONS_FILE")
    QUESTIONS_DEST="${OUTPUT_DIR}/${QUESTIONS_BASENAME}"
    if [[ ! -f "$QUESTIONS_DEST" ]]; then
        mv "$QUESTIONS_FILE" "$QUESTIONS_DEST"
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
python -m prompt_generator.evaluation.parallel_runner \
    "$ANNOTATIONS" \
    "$VIDEO_DIR" \
    -o "$OUTPUT_DIR" \
    -g "$NUM_GPUS" \
    -m "$MODEL" \
    -f "$NUM_FRAMES" \
    --thinking-budget 512 \
    --max-new-tokens 1024 \
    --stagger-delay "$STAGGER_DELAY" \
    $EXTRA_ARGS &

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
