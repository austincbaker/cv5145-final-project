#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "Social Grading: ${MODEL_NAME}"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

. ~/miniconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

conda activate vlm_py312

cd ~/aggressive_behavior_project
export PYTHONPATH="$PWD:$PYTHONPATH"

RESP="${RESP_FILE:-freeform_eval/social_results/social_responses_${MODEL_NAME}.json}"
OUT_BASE=$(basename "$RESP" .json)
OUT="freeform_eval/social_results/grades_${OUT_BASE}.json"

if [ ! -f "$RESP" ]; then
    echo "FATAL: $RESP not found"
    exit 1
fi

echo "=== Grading ${MODEL_NAME} ==="
python freeform_eval/grade_social_appropriateness.py --responses "$RESP" --judge transformers --judge-model Qwen/Qwen2.5-7B-Instruct -o "$OUT"

echo "Grading complete: $(date)"
