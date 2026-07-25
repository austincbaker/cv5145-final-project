#!/bin/bash
# Submit role_graph eval for top 5 models, 4 parts each.
# Run from ~/aggressive_behavior_project on the cluster.
set -e

QUESTIONS="train_model/data/generated_questions.json"
OUTDIR="role_graph_results"
mkdir -p "$OUTDIR"

# Split questions into 4 parts (if not already done)
if [ ! -f "train_model/data/generated_questions_part1of4.json" ]; then
    python split_questions.py "$QUESTIONS" -n 4
fi

# model_name|hf_path|conda_env|gres|mem
MODELS=(
    "InternVL2.5-78B-AWQ|OpenGVLab/InternVL2_5-78B-AWQ|vlm_py312_tf4451|gpu:ampere:1,gpumem:80G|128G"
    "Qwen3-VL-8B|Qwen/Qwen3-VL-8B-Instruct|vlm_py312_fromsrc|gpu:ampere:1,gpumem:48G|48G"
    "InternVL2.5-8B|OpenGVLab/InternVL2_5-8B|vlm_py312|gpu:ampere:1,gpumem:48G|48G"
    "InternVL3-9B|OpenGVLab/InternVL3-9B|vlm_py312|gpu:ampere:1,gpumem:48G|48G"
    "gemma-4-26B|google/gemma-4-26b-a4b-it|vlm_gemma4|gpu:ampere:1,gpumem:80G|128G"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME HF CONDA GRES MEM <<< "$entry"
    echo "=== $NAME ==="
    for P in 1 2 3 4; do
        QFILE="train_model/data/generated_questions_part${P}of4.json"
        JID=$(sbatch --parsable --job-name="rg_${NAME}_p${P}" --output="$OUTDIR/${NAME}_rg_p${P}_%j.out" --error="$OUTDIR/${NAME}_rg_p${P}_%j.err" --partition=gpu --qos=group3 --gres="$GRES" --mem="$MEM" --cpus-per-task=4 --requeue --signal=B:USR1@120 --open-mode=append --wrap=". ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA && cd ~/aggressive_behavior_project && export PYTHONIOENCODING=utf-8 && export PYTHONUNBUFFERED=1 && export PYTHONPATH=\"\$PWD:\$PYTHONPATH\" && python eval_role_graph.py --model $HF --questions-json $QFILE -o $OUTDIR/${NAME}_role_graph_part${P}of4.json")
        echo "  Part $P: $JID"
    done
done
