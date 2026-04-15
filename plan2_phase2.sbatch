#!/bin/bash
#SBATCH --job-name=plan2_phase2_cot
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=plan2_phase2_%j.out
#SBATCH --error=plan2_phase2_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 2: CoT Chain Generation via GPT-4o
# Generates step-by-step reasoning chains for compound/complex questions
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
echo "Plan 2 Phase 2: CoT Chain Generation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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
echo "Phase 2: Generating CoT chains..."
echo "=============================================="
echo ""

# Check for Phase 1 outputs
if [ ! -f "plan2_data/sft_train.json" ]; then
    echo "ERROR: Phase 1 output (plan2_data/sft_train.json) not found"
    echo "Phase 1 must complete before Phase 2"
    exit 1
fi

# Print GPU info
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "Generating reasoning chains for compound/complex questions using InternVL2.5-26B..."
python plan2_cot/generate_cot_chains.py \
    --train-data plan2_data/sft_train.json \
    --output plan2_cot/cot_chains_train.json \
    --model-name OpenGVLab/InternVL2_5-26B \
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
echo "  CoT chains: plan2_cot/cot_chains_train.json"
echo ""
echo "Next: Phase 3 (CoT SFT Training)"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
