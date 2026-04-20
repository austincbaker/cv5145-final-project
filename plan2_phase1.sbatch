#!/bin/bash
#SBATCH --job-name=plan2_phase1_sft
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=plan2_phase1_%j.out
#SBATCH --error=plan2_phase1_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 1: Multimodal SFT Baseline
# Fine-tune InternVL2.5-8B (vision + LLM) on (frames, question) -> answer.
# ==============================================================================

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

handle_timeout_signal() {
    echo ""
    echo "TIMEOUT: USR1 received at $(date); requeueing."
    pkill -P $$
    sleep 5
    scontrol requeue "$SLURM_JOB_ID"
}
trap 'handle_timeout_signal' USR1

cd /home/au182598/bullying-project || exit 1

echo "=============================================="
echo "Plan 2 Phase 1: Multimodal SFT"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)  Requeue: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "Installing dependencies..."
pip install -q -r plan2_requirements.txt

# Bail out if frames aren't cached yet.
if [ ! -d plan2_data/frames ] || [ -z "$(ls -A plan2_data/frames 2>/dev/null)" ]; then
    echo "ERROR: plan2_data/frames is empty. Run plan2_phase0_frames.sbatch first."
    exit 1
fi

echo ""
echo "=============================================="
echo "Training multimodal SFT baseline..."
echo "=============================================="
python plan2_sft/train_sft.py --config plan2_configs/sft.yaml "$@"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "Phase 1 complete: $(date)"
exit 0
