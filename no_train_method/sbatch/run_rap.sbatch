#!/bin/bash
#SBATCH --job-name=rap_1shot
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=no_train_method/rap_%j.out
#SBATCH --error=no_train_method/rap_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"

. ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-vlm_train_py312}"

TRAIN_DATA="${TRAIN_DATA:-train_model/data/generated_questions.json}"
TEST_DATA="${TEST_DATA:-train_model/experiments/5pct_adaptive_v1/data/sft_test.json}"
DATASET_JSON="${DATASET_JSON:-dataset.json}"
OUTPUT="${OUTPUT:-no_train_method/results/rap_eval.json}"
PART="${PART:-}"
TOTAL_PARTS="${TOTAL_PARTS:-}"

PART_ARGS=""
if [ -n "$PART" ] && [ -n "$TOTAL_PARTS" ]; then
    PART_ARGS="--part $PART --total-parts $TOTAL_PARTS"
fi

python no_train_method/run_eval.py \
    --train-data "$TRAIN_DATA" \
    --test-data "$TEST_DATA" \
    --dataset-json "$DATASET_JSON" \
    -o "$OUTPUT" \
    $PART_ARGS
