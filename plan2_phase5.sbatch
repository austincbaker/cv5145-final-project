#!/bin/bash
#SBATCH --job-name=plan2_phase5_adpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=plan2_phase5_%j.out
#SBATCH --error=plan2_phase5_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 5: ADPO Training
# Trains with ADPO loss on preference pairs:
#   L_ADPO = L_preference(chosen > rejected) + α * L_anchor(model | reference)
#
# Where:
#   - L_preference: Bradley-Terry ranking loss
#   - L_anchor: KL divergence from Phase 3 checkpoint
#   - α: anchoring strength (default 0.5)
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
echo "Plan 2 Phase 5: ADPO Training"
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
echo "Phase 5: ADPO Training"
echo "=============================================="
echo ""

# Check for required inputs
if [ ! -f "plan2_models/cot_sft/adapter_config.json" ]; then
    echo "ERROR: Phase 3 checkpoint not found"
    exit 1
fi

if [ ! -f "plan2_adpo/preference_pairs_train.json" ]; then
    echo "ERROR: Phase 4 preference pairs not found"
    exit 1
fi

# Optionally sweep alpha values
echo "Training ADPO with α=0.5 (baseline)..."
python plan2_adpo/train_adpo.py \
    --pairs plan2_adpo/preference_pairs_train.json \
    --reference plan2_models/cot_sft \
    --output plan2_models/adpo_final \
    --alpha 0.5 \
    --epochs 2 \
    --batch-size 8 \
    --lr 1e-4

if [ $? -ne 0 ]; then
    echo "ERROR: ADPO training failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Phase 5 Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Final model: plan2_models/adpo_final/"
echo ""
echo "Next: Phase 6 (Evaluation & Ablation Study)"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
