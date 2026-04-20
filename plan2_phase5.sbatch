#!/bin/bash
#SBATCH --job-name=plan2_phase5_adpo
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=plan2_phase5_%j.out
#SBATCH --error=plan2_phase5_%j.err
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 5: Multimodal (A)DPO training
# Loss: DPO against the Phase 3 adapter as reference (via disable_adapter).
# Pass --override dpo.alpha=0.1 (etc.) for anchor-term sweeps.
# ==============================================================================

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
conda activate base

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
echo "Plan 2 Phase 5: Multimodal (A)DPO"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)  Requeue: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================="

pip install -q -r plan2_requirements.txt

if [ ! -f plan2_adpo/preference_pairs_train.json ]; then
    echo "ERROR: plan2_adpo/preference_pairs_train.json not found. Run Phase 4 first."
    exit 1
fi

python plan2_adpo/train_adpo.py --config plan2_configs/adpo.yaml "$@"

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 5 training failed"
    exit 1
fi

echo "Phase 5 complete: $(date)"
exit 0
