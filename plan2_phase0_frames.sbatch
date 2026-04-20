#!/bin/bash
#SBATCH --job-name=plan2_phase0_frames
#SBATCH --partition=short              # CPU-only, 23h cap — plenty for frame extraction
#SBATCH --qos=group3
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=plan2_phase0_%j.out
#SBATCH --error=plan2_phase0_%j.err
#SBATCH --time=2:00:00

# ==============================================================================
# Plan 2 Phase 0: One-time frame extraction
# Extracts frames_per_video frames (default 8) from every video referenced in
# sft_train/val/test.json and caches them under plan2_data/frames/.
# ==============================================================================

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd /home/au182598/bullying-project || exit 1

echo "=============================================="
echo "Plan 2 Phase 0: Frame extraction"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

python -m plan2_common.frame_cache \
    --config plan2_configs/base.yaml \
    --workers 16

echo "=============================================="
echo "Phase 0 complete: $(date)"
echo "=============================================="
exit 0
