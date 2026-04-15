#!/bin/bash
#SBATCH --job-name=plan2_phase6_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=plan2_phase6_%j.out
#SBATCH --error=plan2_phase6_%j.err
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 6: Full Evaluation & Ablation Study
# Compares all three pipeline stages:
#   1. Phase 1 SFT baseline
#   2. Phase 3 SFT+CoT
#   3. Phase 5 ADPO final
# ==============================================================================

# Environment setup
eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh

module use ~/privatemodules
module load anaconda/25.11.1
conda activate base
module load cuda/12.6

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
echo "Plan 2 Phase 6: Evaluation & Ablation Study"
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
echo "Phase 6: Full Evaluation"
echo "=============================================="
echo ""

# Check for test data
if [ ! -f "plan2_data/sft_test.json" ]; then
    echo "ERROR: Test data not found"
    exit 1
fi

echo "Evaluating all pipeline stages..."
python plan2_eval/run_full_evaluation.py \
    --sft plan2_models/sft_baseline \
    --cot plan2_models/cot_sft \
    --adpo plan2_models/adpo_final \
    --test-data plan2_data/sft_test.json \
    --output plan2_eval/full_evaluation_results.json

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Phase 6 Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Results: plan2_eval/full_evaluation_results.json"
echo ""
echo "Results Summary:"
if [ -f "plan2_eval/full_evaluation_results.json" ]; then
    cat plan2_eval/full_evaluation_results.json | python -m json.tool | head -50
fi
echo ""
echo "=============================================="
echo "FULL PIPELINE COMPLETE!"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
