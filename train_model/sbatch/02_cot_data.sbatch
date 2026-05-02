#!/bin/bash
#SBATCH --job-name=train_model_02_cot
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=train_model_02_%j.out
#SBATCH --error=train_model_02_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Phase 2: CoT Chain Generation via GPT-4o
# Generates step-by-step reasoning chains for compound/complex questions
# ==============================================================================

# Environment setup
eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh

conda activate vlm_train_py312

# Add repo to Python path for imports
export PYTHONPATH="/home/au182598/aggressive_behavior_project:${PYTHONPATH}"

# Signal handler
handle_timeout_signal() {
    echo ""
    echo "=============================================="
    echo "TIMEOUT WARNING: Received USR1 signal"
    echo "Job will be requeued and resume from checkpoint"
    echo "Terminating gracefully..."
    echo "Time: $(date)"
    echo "=============================================="
    pkill -P $$
    sleep 5
    scontrol requeue "$SLURM_JOB_ID"
}

trap 'handle_timeout_signal' USR1

echo "=============================================="
echo "Phase 2: CoT Chain Generation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="
echo ""

cd /home/au182598/aggressive_behavior_project || exit 1

# Force unbuffered Python output so SLURM .out files update in real time
export PYTHONUNBUFFERED=1

# Install dependencies
echo "Installing dependencies..."
pip install -q -r train_model/requirements.txt

echo ""
echo "=============================================="
echo "Phase 2: Generating CoT chains..."
echo "=============================================="
echo ""

# Check for Phase 1 outputs
if [ ! -f "train_model/data/sft_train.json" ]; then
    echo "ERROR: Phase 1 output (train_model/data/sft_train.json) not found"
    echo "Phase 1 must complete before Phase 2"
    exit 1
fi

# Print GPU info
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "Generating reasoning chains for compound/complex questions using InternVL2.5-8B..."
python train_model/cot/generate_chains.py \
    --train-data train_model/data/sft_train.json \
    --output train_model/data/cot_chains.json \
    --model-name OpenGVLab/InternVL2_5-8B \
    --sample-rate 1.0

if [ $? -ne 0 ]; then
    echo "ERROR: CoT generation failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Phase 2 Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  CoT chains: train_model/data/cot_chains.json"
echo ""
echo "Next: Phase 3 (CoT SFT Training)"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
