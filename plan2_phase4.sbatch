#!/bin/bash
#SBATCH --job-name=plan2_phase4_adpo_pairs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=plan2_phase4_%j.out
#SBATCH --error=plan2_phase4_%j.err
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 4: Extract ADPO Preference Pairs
# Extracts preference pairs from improved distractor pipeline:
#   Chosen: correct answer (with CoT for compounds if available)
#   Rejected: hard negatives ranked by hardness type
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
echo "Plan 2 Phase 4: ADPO Preference Pair Extraction"
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
echo "Phase 4: Extracting preference pairs..."
echo "=============================================="
echo ""

# Check for Phase 1 output
if [ ! -f "plan2_data/generated_questions_plan2.json" ]; then
    echo "ERROR: Phase 1 questions not found"
    exit 1
fi

# Phase 2 CoT chains are optional
if [ ! -f "plan2_cot/cot_chains_train.json" ]; then
    echo "Warning: CoT chains not found, preference pairs will use direct answers"
fi

echo "Extracting preference pairs from improved distractor pipeline..."
python plan2_adpo/extract_preference_pairs.py \
    --questions plan2_data/generated_questions_plan2.json \
    --cot-chains plan2_cot/cot_chains_train.json \
    --output plan2_adpo/preference_pairs_train.json \
    --num-rejected 3

if [ $? -ne 0 ]; then
    echo "ERROR: Preference pair extraction failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Phase 4 Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Preference pairs: plan2_adpo/preference_pairs_train.json"
echo ""
echo "Next: Phase 5 (ADPO Training)"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
