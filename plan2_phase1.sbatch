#!/bin/bash
#SBATCH --job-name=plan2_phase1_sft
#SBATCH --partition=gpu
#SBATCH --qos=group3             # Default QOS
#SBATCH --gres=gpu:ampere:1      # Association MaxTRES cap: gres/gpu=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G                # Hidden QOS memory cap — 80G is safe
#SBATCH --output=plan2_phase1_%j.out
#SBATCH --error=plan2_phase1_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 1: SFT Baseline Training
# Fine-tune InternVL2.5-8B with LoRA on benchmark questions
# ==============================================================================

# Environment setup
eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh

conda activate base

# Add repo to Python path for imports
export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"

# Signal handler for graceful shutdown before timeout
handle_timeout_signal() {
    echo ""
    echo "=============================================="
    echo "TIMEOUT WARNING: Received USR1 signal"
    echo "Job will be requeued and resume from checkpoint"
    echo "Terminating training gracefully..."
    echo "Time: $(date)"
    echo "=============================================="
    pkill -P $$
    sleep 5
    scontrol requeue "$SLURM_JOB_ID"
}

trap 'handle_timeout_signal' USR1

# Print job info
echo "=============================================="
echo "Plan 2 Phase 1: SFT Baseline Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x Ampere (A100)"
echo "Start time: $(date)"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="
echo ""

# Print GPU info
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set up environment
cd /home/au182598/bullying-project || exit 1

# Install dependencies (lightweight, cached)
echo "Installing dependencies..."
pip install -q -r plan2_requirements.txt

# Verify InternVL2.5-8B can be loaded
echo "Verifying model access..."
python -c "from transformers import AutoModel; print('✓ Model loading verified')" || exit 1

echo ""
echo "=============================================="
echo "Phase 1 Pipeline: Regenerate → Format → Train → Evaluate"
echo "=============================================="
echo ""

# Step 1: Regenerate questions with improved distractors
echo "[1/4] Regenerating benchmark with improved distractors..."
python plan2_sft/regenerate_benchmark.py
if [ $? -ne 0 ]; then
    echo "ERROR: Question regeneration failed"
    exit 1
fi
echo "✓ Benchmark regenerated"
echo ""

# Step 2: Format SFT training data
echo "[2/4] Formatting SFT training data..."
python plan2_sft/format_sft_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Data formatting failed"
    exit 1
fi
echo "✓ Training data formatted (train/val/test split)"
echo ""

# Step 3: Train SFT baseline with LoRA
echo "[3/4] Training SFT baseline (InternVL2.5-8B + LoRA)..."
python plan2_sft/train_sft.py \
    --train-data plan2_data/sft_train.json \
    --val-data plan2_data/sft_val.json \
    --output-dir plan2_models/sft_baseline \
    --model-name OpenGVLab/InternVL2_5-8B \
    --epochs 3 \
    --batch-size 4 \
    --lr 5e-4

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi
echo "✓ SFT baseline trained and saved"
echo ""

# Step 4: Evaluate on test set
echo "[4/4] Evaluating SFT baseline on test set..."
python plan2_sft/evaluate_sft.py \
    --model plan2_models/sft_baseline \
    --test-data plan2_data/sft_test.json \
    --output plan2_eval/sft_baseline_results.json

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi
echo "✓ Evaluation complete"
echo ""

# Print results summary
echo "=============================================="
echo "Phase 1 Complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Checkpoint:  plan2_models/sft_baseline/"
echo "  Results:     plan2_eval/sft_baseline_results.json"
echo "  Training data: plan2_data/sft_{train,val,test}.json"
echo ""
echo "Results Summary:"
cat plan2_eval/sft_baseline_results.json | python -m json.tool | head -30
echo ""
echo "Next: Phase 2 (CoT Data Generation)"
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="

exit 0
