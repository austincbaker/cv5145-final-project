#!/bin/bash
#SBATCH --job-name=counting_social
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=freeform_eval/counting_social_%j.out
#SBATCH --error=freeform_eval/counting_social_%j.err

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
conda activate vlm_py312

export PYTHONPATH="/home/au182598/aggressive_behavior_project:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd /home/au182598/aggressive_behavior_project || exit 1

MODEL="${MODEL:?Set MODEL env var}"
OUTPUT="${OUTPUT:?Set OUTPUT env var}"
CONDA_ENV="${CONDA_ENV:-vlm_py312}"

conda activate "$CONDA_ENV"

echo "=============================================="
echo "Counting + Social Appropriateness Eval"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

python freeform_eval/run_counting_social.py \
    --eval-file freeform_eval/counting_social_eval.json \
    --model "$MODEL" \
    --num-frames 8 \
    -o "$OUTPUT"

echo "=============================================="
echo "Done: $(date)"
echo "=============================================="
