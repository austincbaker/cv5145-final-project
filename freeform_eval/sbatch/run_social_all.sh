#!/bin/bash
# Submit social appropriateness eval for top 5 models + gpt-oss grading.
#
# Pipeline:
#   Step 1: 5 parallel VLM response jobs (each model on all 2687 videos)
#   Step 2: Install gpt-oss vLLM environment (after all 5 finish)
#   Step 3: Grade all 5 result files with gpt-oss-20b
#
# Usage:
#   bash freeform_eval/sbatch/run_social_all.sh

set -e
cd "$(dirname "$0")/../.."

DIR="freeform_eval/sbatch"

# Top 5 models: name|hf_path|conda_env|gres|mem
declare -a MODELS=(
    "InternVL2.5-78B-AWQ|OpenGVLab/InternVL2_5-78B-AWQ|vlm_py312_tf4451|gpu:ampere:1,gpumem:80G|128G"
    "InternVL2.5-8B|OpenGVLab/InternVL2_5-8B|vlm_py312|gpu:ampere:1,gpumem:48G|48G"
    "Qwen3-VL-8B|Qwen/Qwen3-VL-8B-Instruct|vlm_py312_fromsrc|gpu:ampere:1,gpumem:48G|48G"
    "Ovis2.5-9B-Thinking|AIDC-AI/Ovis2.5-9B|vlm_py312|gpu:ampere:1,gpumem:48G|48G"
    "Ovis2.5-9B|AIDC-AI/Ovis2.5-9B|vlm_py312|gpu:ampere:1,gpumem:48G|48G"
)

JOB_IDS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME HF_PATH CONDA GRES MEM <<< "$entry"
    OUTFILE="freeform_eval/social_results/social_responses_${NAME}.json"

    JOB_ID=$(sbatch --parsable \
        --job-name="social_${NAME}" \
        --output="freeform_eval/social_results/social_${NAME}_%j.out" \
        --error="freeform_eval/social_results/social_${NAME}_%j.err" \
        --partition=gpu \
        --gres="${GRES}" \
        --mem="${MEM}" \
        --cpus-per-task=4 \
        --wrap=". ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA} && cd ~/aggressive_behavior_project && export PYTHONPATH=\"\$PWD:\$PYTHONPATH\" && python freeform_eval/run_social_appropriateness.py --model ${HF_PATH} --annotations annotations.json --max-new-tokens 512 -o ${OUTFILE}")

    JOB_IDS+=("$JOB_ID")
    echo "Submitted ${NAME}: ${JOB_ID}"
done

# Build dependency string for grading job
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

# Grading job: starts vLLM with gpt-oss-20b, grades all 5 result files
GRADE_JOB=$(sbatch --parsable \
    --dependency=afterok:${DEP_STR} \
    "${DIR}/grade_social_gptoss.sbatch")

echo "Submitted grading: ${GRADE_JOB} (depends on ${DEP_STR})"

echo ""
echo "Pipeline submitted:"
for i in "${!MODELS[@]}"; do
    IFS='|' read -r NAME _ _ _ _ <<< "${MODELS[$i]}"
    echo "  ${NAME}: ${JOB_IDS[$i]}"
done
echo "  Grading: ${GRADE_JOB}"
echo ""
echo "Monitor: squeue -u \$USER"
