#!/bin/bash
#SBATCH --job-name=eval_multi_gpu
#SBATCH --account=yrawat
#SBATCH --qos=yrawat
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --output=eval_multi_%j.out
#SBATCH --error=eval_multi_%j.err
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
module load anaconda/anaconda-2024.10
conda activate ovis2_5
module load cuda/cuda-12.4.0

# Configuration
ANNOTATIONS="annotations.json"
VIDEO_DIR="videos/"
MODEL="${MODEL:-ovis-9b}"
QUESTIONS_FILE="${QUESTIONS_FILE:-}"
NUM_FRAMES=8
NUM_GPUS=2
STAGGER_DELAY=30

# Per-model output directory so results don't collide
MODEL_TAG="${MODEL//\//_}"
OUTPUT_DIR="./results_${MODEL_TAG}"

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
