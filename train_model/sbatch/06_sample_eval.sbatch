#!/bin/bash
#SBATCH --job-name=sample_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "Sample eval: max_new_tokens=192 on 100 examples"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)  Requeue: $SLURM_RESTART_COUNT"
echo "=============================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_train_py312

cd ~/aggressive_behavior_project
export PYTHONPATH="$PWD:$PYTHONPATH"

python train_model/eval/run_evaluation.py --config train_model/configs/eval.yaml --override generation.max_new_tokens=192 --override data.test=train_model/data/sft_test_sample100.json --override output=train_model/eval/results_sample192.json --override stages.adpo=train_model/models/dpo

echo "Sample eval complete: $(date)"
