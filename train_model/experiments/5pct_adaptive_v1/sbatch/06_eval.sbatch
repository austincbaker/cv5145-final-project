#!/bin/bash
#SBATCH --job-name=5pct_av1_eval
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=train_model/experiments/5pct_adaptive_v1/logs/%x_%j.out
#SBATCH --error=train_model/experiments/5pct_adaptive_v1/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "5pct_adaptive_v1: Phase 6 -- Final eval (SFT, CoT, ADPO)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Args: $@"
echo "Start: $(date)"
echo "=============================================="

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

python train_model/eval/run_evaluation.py --config train_model/experiments/5pct_adaptive_v1/configs/eval.yaml "$@"

echo "Final eval complete: $(date)"
