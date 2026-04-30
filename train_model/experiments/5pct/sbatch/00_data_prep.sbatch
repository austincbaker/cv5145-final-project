#!/bin/bash
#SBATCH --job-name=5pct_data
#SBATCH --output=train_model/experiments/5pct/logs/%x_%j.out
#SBATCH --error=train_model/experiments/5pct/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "5% Baseline: Data Preparation"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312
cd ~/aggressive_behavior_project
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== Step 1: Format 5% split ==="
python train_model/sft/format_data.py --questions train_model/data/generated_questions.json --annotations annotations.json --groups dataset.json --output-dir train_model/experiments/5pct/data --train-ratio 0.05 --val-ratio 0.0 --force

echo "=== Step 2: CoT distillation ==="
python train_model/cot/generate_chains.py --train-data train_model/experiments/5pct/data/sft_train.json --output train_model/experiments/5pct/data/cot_chains.json

echo "=== Step 3: Preference pair extraction ==="
python train_model/dpo/extract_pairs.py --questions train_model/data/generated_questions.json --cot-chains train_model/experiments/5pct/data/cot_chains.json --output train_model/experiments/5pct/data/preference_pairs.json --train-split train_model/experiments/5pct/data/sft_train.json --annotations annotations.json

echo "Data prep complete: $(date)"
