#!/bin/bash
#SBATCH --job-name=train_model_05_adpo
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=train_model_05_%j.out
#SBATCH --error=train_model_05_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Phase 5: Multimodal DPO training
# Loss: DPO against the Phase 3 adapter as reference (via disable_adapter).
# Pass --override dpo.alpha=0.1 (etc.) for anchor-term sweeps.
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
echo "Phase 5: Multimodal DPO"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)  Requeue: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="

pip install -q -r train_model/requirements.txt

if [ ! -f train_model/data/preference_pairs.json ]; then
    echo "ERROR: train_model/data/preference_pairs.json not found. Run Phase 4 first."
    exit 1
fi

python train_model/dpo/train.py --config train_model/configs/dpo.yaml "$@"

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 5 training failed"
    exit 1
fi

echo "Phase 5 complete: $(date)"
exit 0
