#!/bin/bash
#SBATCH --job-name=plan2_smoke
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=plan2_smoke_%j.out
#SBATCH --error=plan2_smoke_%j.err
#SBATCH --time=2:00:00

# ==============================================================================
# Plan 2 Phase 1 SMOKE TEST — tiny multimodal training run to catch bugs before
# committing the full 24h slot. Uses sft_val.json as a tiny train set, 100
# training steps total, writes to plan2_models/sft_smoke/.
# ==============================================================================

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd /home/au182598/bullying-project || exit 1

echo "=============================================="
echo "Plan 2 SMOKE TEST (Phase 1 multimodal)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

pip install -q -r plan2_requirements.txt

if [ ! -d plan2_data/frames ] || [ -z "$(ls -A plan2_data/frames 2>/dev/null)" ]; then
    echo "ERROR: plan2_data/frames is empty. Run plan2_phase0_frames.sbatch first."
    exit 1
fi

python plan2_sft/train_sft.py --config plan2_configs/sft_smoke.yaml "$@"

echo "Smoke test done: $(date)"
exit 0
