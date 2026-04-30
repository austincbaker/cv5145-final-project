#!/bin/bash
#SBATCH --job-name=5pct_adpo
#SBATCH --output=train_model/experiments/5pct/logs/%x_%j.out
#SBATCH --error=train_model/experiments/5pct/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "5% Baseline: ADPO Only (no SFT warmup, alpha=0.5)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONPATH="$PWD:$PYTHONPATH"

python train_model/dpo/train.py --config train_model/experiments/5pct/configs/adpo.yaml

echo "ADPO complete: $(date)"
