#!/bin/bash
#SBATCH --job-name=ovis_merge
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --output=ovis_role_graph_merge_%j.out
#SBATCH --error=ovis_role_graph_merge_%j.err
#SBATCH --open-mode=append

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

. ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm_py312

cd ~/aggressive_behavior_project

echo "Merging role_graph results: $(date)"

PART0="ovis2_5_9B_generated_questions_from0_to667_role_graph.json"
PART1="ovis2_5_9B_generated_questions_from667_to1334_role_graph.json"
PART2="ovis2_5_9B_generated_questions_from1334_to2001_role_graph.json"
PART3="ovis2_5_9B_generated_questions_from2001_role_graph.json"

for f in "$PART0" "$PART1" "$PART2" "$PART3"; do
    if [ ! -f "$f" ]; then echo "MISSING: $f"; exit 1; fi
    echo "Found: $f ($(python -c "import json; print(json.load(open('$f',encoding='utf-8'))['total_questions'])" 2>/dev/null) questions)"
done

python ~/cv-research/bullying-project/analysis_scripts/merge_eval_results.py --eval "$PART1" "$PART2" "$PART3" --combined "$PART0"

mv "$PART0" ovis2_5_9B_role_graph_combined.json
echo "Merged to: ovis2_5_9B_role_graph_combined.json"
echo "Done: $(date)"
