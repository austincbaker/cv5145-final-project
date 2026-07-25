#!/bin/bash
#SBATCH --job-name=20pct_easy_pairs
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=train_model/experiments/20pct_easy_v1/logs/%x_%j.out
#SBATCH --error=train_model/experiments/20pct_easy_v1/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

echo '=============================================='
echo '20pct_easy_v1: Phase 4 -- Extract easy-mining preference pairs'
echo 'Job ID: '$SLURM_JOB_ID'  Node: '$SLURM_NODELIST
echo 'Start: 'Sat May  2 19:37:56 EDT 2026
echo '=============================================='

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

python train_model/dpo/extract_pairs.py --questions train_model/data/generated_questions.json --cot-chains train_model/experiments/20pct_adaptive_v1/data/cot_chains.json --output train_model/experiments/20pct_easy_v1/data/preference_pairs_easy.json --train-split train_model/experiments/20pct_adaptive_v1/data/sft_train.json --annotations annotations.json --eval-results train_model/experiments/20pct_adaptive_v1/data/eval_train_cot.json --selection-strategy easy_mining

echo 'Easy-mining pair extraction complete: 'Sat May  2 19:37:56 EDT 2026
