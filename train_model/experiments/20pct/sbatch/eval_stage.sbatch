#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=group3
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

echo "20pct standalone eval -- Stage: $EVAL_STAGE -- Job $SLURM_JOB_ID -- Args: $@ -- $(date)"
python train_model/eval/run_evaluation.py --config "train_model/experiments/20pct/configs/eval_${EVAL_STAGE}.yaml" "$@"
echo "Eval complete: $(date)"
