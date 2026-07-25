#!/bin/bash
#SBATCH --job-name=eval_adpo
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=evals/adpo/logs/%x_%j.out
#SBATCH --error=evals/adpo/logs/%x_%j.err
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

python train_model/eval/run_evaluation.py --config evals/adpo/eval.yaml "$@"
