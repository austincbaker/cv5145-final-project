#!/bin/bash
#SBATCH --job-name=5pct_as_adpo
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=train_model/experiments/5pct_adaptive_standalone/logs/%x_%j.out
#SBATCH --error=train_model/experiments/5pct_adaptive_standalone/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "5pct adaptive standalone ADPO -- Job $SLURM_JOB_ID -- $(date)"
python train_model/dpo/train.py --config train_model/experiments/5pct_adaptive_standalone/configs/adpo.yaml
echo "ADPO complete: $(date)"
