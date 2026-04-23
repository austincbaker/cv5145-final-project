#!/bin/bash
#SBATCH --job-name=train_model_01_sft
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --output=train_model_01_%j.out
#SBATCH --error=train_model_01_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Phase 1: Multimodal SFT
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
echo "Phase 1: Multimodal SFT"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)  Requeue: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "Installing dependencies..."
pip install -q -r train_model/requirements.txt

# Bail out if frames aren't cached yet.
if [ ! -d train_model/data/frames ] || [ -z "$(ls -A train_model/data/frames 2>/dev/null)" ]; then
    echo "ERROR: train_model/data/frames is empty. Run train_model/sbatch/00_extract_frames.sbatch first."
    exit 1
fi

echo ""
echo "=============================================="
echo "Training multimodal SFT baseline..."
echo "=============================================="
python train_model/sft/train.py --config train_model/configs/sft.yaml "$@"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "Phase 1 complete: $(date)"
exit 0
