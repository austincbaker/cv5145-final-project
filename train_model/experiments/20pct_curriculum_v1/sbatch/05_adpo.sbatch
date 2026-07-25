#!/bin/bash
#SBATCH --job-name=20pct_cur_adpo
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=train_model/experiments/20pct_curriculum_v1/logs/%x_%j.out
#SBATCH --error=train_model/experiments/20pct_curriculum_v1/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

echo "==============================================" 
echo "20pct_curriculum_v1: Phase 5 -- ADPO (curriculum pairs, resumes from CoT-SFT)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

python train_model/dpo/train.py --config train_model/experiments/20pct_curriculum_v1/configs/adpo.yaml

echo "ADPO (curriculum) complete: $(date)"
