#!/bin/bash
#SBATCH --job-name=eval_sft
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=evals/sft/logs/%x_%j.out
#SBATCH --error=evals/sft/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

echo '=============================================='
echo 'Standalone SFT eval'
echo 'Job ID: '$SLURM_JOB_ID'  Node: '$SLURM_NODELIST
echo 'Start: 'Sat May  2 23:45:31 EDT 2026
echo '=============================================='

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

python train_model/eval/run_evaluation.py --config evals/sft/eval.yaml "$@"

echo 'Eval complete: 'Sat May  2 23:45:31 EDT 2026
