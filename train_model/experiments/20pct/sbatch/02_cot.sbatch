#!/bin/bash
#SBATCH --job-name=20pct_cot
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=train_model/experiments/20pct/logs/%x_%j.out
#SBATCH --error=train_model/experiments/20pct/logs/%x_%j.err
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

echo "20pct standalone CoT -- Job $SLURM_JOB_ID -- $(date)"
python train_model/cot/train.py --config train_model/experiments/20pct/configs/cot.yaml
echo "CoT complete: $(date)"
