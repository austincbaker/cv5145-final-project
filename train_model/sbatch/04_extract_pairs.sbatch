#!/bin/bash
#SBATCH --job-name=train_model_04_adpo_pairs
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=train_model_04_%j.out
#SBATCH --error=train_model_04_%j.err
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Phase 4: Extract Preference Pairs
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
echo "Phase 4: Preference Pair Extraction"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="
echo ""

cd /home/au182598/bullying-project || exit 1

# Force unbuffered Python output so SLURM .out files update in real time
export PYTHONUNBUFFERED=1

# Install dependencies
echo "Installing dependencies..."
pip install -q -r train_model/requirements.txt

echo ""
echo "=============================================="
echo "Phase 4: Extracting preference pairs..."
echo "=============================================="
echo ""

# Check for Phase 1 output
if [ ! -f "train_model/data/generated_questions.json" ]; then
    echo "ERROR: Phase 1 questions not found"
    exit 1
fi

# Phase 2 CoT chains are optional
if [ ! -f "train_model/data/cot_chains.json" ]; then
    echo "Warning: CoT chains not found, preference pairs will use direct answers"
fi

echo "Extracting preference pairs from improved distractor pipeline..."
python train_model/dpo/extract_pairs.py \
    --questions train_model/data/generated_questions.json \
    --cot-chains train_model/data/cot_chains.json \
    --output train_model/data/preference_pairs.json \
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
echo "  Preference pairs: train_model/data/preference_pairs.json"
echo ""
echo "Next: Phase 5 (ADPO Training)"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
