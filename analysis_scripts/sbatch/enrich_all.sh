#!/bin/bash
# Submit annotation enrichment using InternVL2.5-78B-AWQ.
# Processes all 2687 videos to gather demographic and environmental metadata.
#
# Usage:
#   bash analysis_scripts/sbatch/enrich_all.sh

set -e
cd "$(dirname "$0")/../.."

mkdir -p analysis_scripts/output/enrichment

sbatch --parsable \
    --job-name="enrich_78B" \
    --output="analysis_scripts/output/enrichment/enrich_78B_%j.out" \
    --error="analysis_scripts/output/enrichment/enrich_78B_%j.err" \
    --partition=gpu \
    --gres=gpu:ampere:1,gpumem:80G \
    --mem=128G \
    --cpus-per-task=4 \
    --wrap=". ~/miniconda3/etc/profile.d/conda.sh && conda activate vlm_py312_tf4451 && cd ~/aggressive_behavior_project && export PYTHONPATH=\"\$PWD:\$PYTHONPATH\" && export PYTHONIOENCODING=utf-8 && python analysis_scripts/enrich_annotations.py --model OpenGVLab/InternVL2_5-78B-AWQ --annotations annotations.json -o analysis_scripts/output/enrichment/enriched_InternVL2.5-78B-AWQ.json"

echo "Monitor: squeue -u \$USER"
