#!/bin/bash
#SBATCH --job-name=bullying_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
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
QUESTIONS_FILE="${QUESTIONS_FILE:-}"
NUM_FRAMES="${NUM_FRAMES:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
STAGGER_DELAY=30

# Per-run output directory: model tag + job ID so multiple runs don't collide,
# and the directory stays stable across requeues of the same job.
MODEL_TAG="${MODEL//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_${MODEL_TAG}-${SLURM_JOB_ID}}"

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

# =============================================================================
# Dependency version management
# =============================================================================
# Optional overrides via --export:
#   TORCH=<version>          exact torch version to install (prefix-matched
#                            against current so "2.1.0" accepts "2.1.0+cu121").
#                            If unset, torch is left alone.
#   TORCH_INDEX=<url>        optional --index-url for the torch install
#                            (e.g. https://download.pytorch.org/whl/cu121).
#   TRANSFORMERS=<version>   exact transformers version. If unset, falls back
#                            to the model-name heuristic below.
# A torch install failure aborts the job (exit 1) because limping along with
# the wrong torch/CUDA combination will produce obscure runtime errors.
# =============================================================================

# --- Torch ------------------------------------------------------------------
if [[ -n "${TORCH:-}" ]]; then
    CURRENT_TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "MISSING")
    echo "Current torch version:  $CURRENT_TORCH_VERSION"
    echo "Required torch version: $TORCH"
    # Prefix match so "2.1.0" accepts "2.1.0+cu121".
    if [[ "$CURRENT_TORCH_VERSION" != "$TORCH"* ]]; then
        echo "Switching torch to $TORCH..."
        TORCH_INSTALL_ARGS=(torch=="$TORCH")
        if [[ -n "${TORCH_INDEX:-}" ]]; then
            TORCH_INSTALL_ARGS+=(--index-url "$TORCH_INDEX")
            echo "Using torch index URL: $TORCH_INDEX"
        fi
        if ! pip install "${TORCH_INSTALL_ARGS[@]}"; then
            echo ""
            echo "=============================================="
            echo "FATAL: failed to install torch==$TORCH"
            if [[ -n "${TORCH_INDEX:-}" ]]; then
                echo "  (via --index-url $TORCH_INDEX)"
            fi
            echo "Aborting the job so we don't silently run against the wrong torch."
            echo "Check pip's output above for the reason; common causes:"
            echo "  * version not available for this Python / CUDA combination"
            echo "  * missing or mistyped TORCH_INDEX for a CUDA-specific wheel"
            echo "=============================================="
            exit 1
        fi
        echo "Installed torch version: $(python -c 'import torch; print(torch.__version__)')"
    else
        echo "Correct torch version already installed"
    fi
fi

# --- Transformers -----------------------------------------------------------
CURRENT_TF_VERSION=$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo "MISSING")
echo "Current transformers version: $CURRENT_TF_VERSION"

if [[ -n "${TRANSFORMERS:-}" ]]; then
    # Explicit override — use it verbatim.
    REQUIRED_TF_VERSION="$TRANSFORMERS"
    echo "Using TRANSFORMERS env override: $REQUIRED_TF_VERSION"
else
    # Model-name heuristic fallback.
    if [[ "$MODEL" == *ovis* ]]; then
        REQUIRED_TF_VERSION="4.57.1"
    elif [[ "$MODEL" == *internvl* || "$MODEL" == *llava* ]]; then
        REQUIRED_TF_VERSION="4.49.0"
    elif [[ "$MODEL" == *videollama* ]]; then
        REQUIRED_TF_VERSION="4.49.0"
    elif [[ "$MODEL" == *internvl3* ]]; then
        REQUIRED_TF_VERSION="4.52.1"
    else
        REQUIRED_TF_VERSION="5.3.0"
    fi
fi

if [[ "$CURRENT_TF_VERSION" != "$REQUIRED_TF_VERSION" ]]; then
    echo "Switching transformers to $REQUIRED_TF_VERSION..."
    if ! (pip uninstall transformers -y && pip install "transformers==$REQUIRED_TF_VERSION"); then
        echo ""
        echo "=============================================="
        echo "FATAL: failed to install transformers==$REQUIRED_TF_VERSION"
        echo "Aborting the job so we don't run against the wrong transformers version."
        echo "=============================================="
        exit 1
    fi
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
