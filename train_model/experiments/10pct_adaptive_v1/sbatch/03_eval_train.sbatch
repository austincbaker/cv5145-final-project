#!/bin/bash
#SBATCH --job-name=10pct_av1_evtrain
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=train_model/experiments/10pct_adaptive_v1/logs/%x_%j.out
#SBATCH --error=train_model/experiments/10pct_adaptive_v1/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "10pct_adaptive_v1: Phase 3.5 -- Eval CoT-SFT on training questions"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

python train_model/eval/run_evaluation.py --config train_model/experiments/10pct_adaptive_v1/configs/eval_train.yaml

echo "Eval-on-train complete: $(date)"
