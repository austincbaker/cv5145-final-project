#!/bin/bash
#SBATCH --job-name=ovis_role_graph
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --output=ovis_role_graph_%j.out
#SBATCH --error=ovis_role_graph_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_py312

echo "Part: START_INDEX=$START_INDEX END_INDEX=$END_INDEX"
echo "Job ID: $SLURM_JOB_ID"

cd ~/aggressive_behavior_project

END_ARG=""
if [ -n "$END_INDEX" ]; then END_ARG="--end_index $END_INDEX"; fi

python test_aggressive_solution.py --questions_json train_model/data/generated_questions.json --start_index $START_INDEX $END_ARG --template role_graph

echo "Part complete: $(date)"
