#!/bin/bash
#SBATCH --job-name=plan2_phase3_cot_sft
#SBATCH --partition=gpu
#SBATCH --qos=group3-1
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=plan2_phase3_%j.out
#SBATCH --error=plan2_phase3_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 3: CoT SFT Training
# Fine-tune Phase 1 checkpoint on mixed dataset:
#   - Compound questions: use reasoning chains
#   - Simple questions: direct answers (no CoT overhead)
# ==============================================================================

# Environment setup
eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh

conda activate base

# Add repo to Python path for imports
export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"

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
echo "Plan 2 Phase 3: CoT SFT Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 1x Ampere (A100)"
echo "Start time: $(date)"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="
echo ""

cd /home/au182598/bullying-project || exit 1

# Install dependencies
echo "Installing dependencies..."
pip install -q -r plan2_requirements.txt

echo ""
echo "=============================================="
echo "Phase 3: CoT SFT Training"
echo "=============================================="
echo ""

# Check for Phase 1 and Phase 2 outputs
if [ ! -f "plan2_models/sft_baseline/adapter_config.json" ]; then
    echo "ERROR: Phase 1 checkpoint not found"
    exit 1
fi

if [ ! -f "plan2_cot/cot_chains_train.json" ]; then
    echo "ERROR: Phase 2 CoT chains not found"
    exit 1
fi

echo "Training on mixed dataset (compounds with CoT, simple without)..."
python plan2_cot/train_cot_sft.py \
    --train-data plan2_cot/cot_chains_train.json \
    --val-data plan2_data/sft_val.json \
    --checkpoint plan2_models/sft_baseline \
    --output-dir plan2_models/cot_sft \
    --model-name OpenGVLab/InternVL2_5-8B \
    --epochs 2 \
    --batch-size 4 \
    --lr 2e-4

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 3 training failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Phase 3 Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Checkpoint: plan2_models/cot_sft/"
echo ""
echo "Next: Phase 4 (ADPO Preference Pair Extraction)"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
