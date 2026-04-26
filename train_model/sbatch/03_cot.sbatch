#!/bin/bash
#SBATCH --job-name=train_model_03_cot_sft
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=train_model_03_%j.out
#SBATCH --error=train_model_03_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Phase 3: Multimodal CoT-SFT
# Fine-tune Phase 1 adapter with reasoning chains on compound questions.
# ==============================================================================

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312

export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

handle_timeout_signal() {
    echo "TIMEOUT: USR1 received at $(date); requeueing."
    pkill -P $$
    sleep 5
    scontrol requeue "$SLURM_JOB_ID"
}
trap 'handle_timeout_signal' USR1

cd /home/au182598/bullying-project || exit 1

echo "=============================================="
echo "Phase 3: Multimodal CoT-SFT"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)  Requeue: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="

pip install -q -r train_model/requirements.txt

if [ ! -f train_model/data/cot_chains.json ]; then
    echo "ERROR: train_model/data/cot_chains.json not found. Run Phase 2 first."
    exit 1
fi

python train_model/cot/train.py --config train_model/configs/cot.yaml "$@"

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 3 training failed"
    exit 1
fi

echo "Phase 3 complete: $(date)"
exit 0
