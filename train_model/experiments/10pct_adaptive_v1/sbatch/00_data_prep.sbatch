#!/bin/bash
#SBATCH --job-name=10pct_av1_data
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
echo "10pct_adaptive_v1: Data Preparation"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== Step 1: Format 10% split ==="
python train_model/sft/format_data.py --questions train_model/data/generated_questions.json --annotations annotations.json --groups dataset.json --output-dir train_model/experiments/10pct_adaptive_v1/data --train-ratio 0.05 --val-ratio 0.0 --force

echo "=== Step 2: CoT distillation ==="
python train_model/cot/generate_chains.py --train-data train_model/experiments/10pct_adaptive_v1/data/sft_train.json --output train_model/experiments/10pct_adaptive_v1/data/cot_chains.json

echo "=== Step 3: Fixed preference pair extraction (pre-adaptive baseline) ==="
python train_model/dpo/extract_pairs.py --questions train_model/data/generated_questions.json --cot-chains train_model/experiments/10pct_adaptive_v1/data/cot_chains.json --output train_model/experiments/10pct_adaptive_v1/data/preference_pairs_fixed.json --train-split train_model/experiments/10pct_adaptive_v1/data/sft_train.json --annotations annotations.json

echo "Data prep complete: $(date)"
