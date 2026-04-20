#!/bin/bash
#SBATCH --job-name=plan2_phase6_eval
#SBATCH --partition=gpu
#SBATCH --qos=group3
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=plan2_phase6_%j.out
#SBATCH --error=plan2_phase6_%j.err
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ==============================================================================
# Plan 2 Phase 6: Multimodal evaluation across all pipeline stages.
# ==============================================================================

eval "$('/home/au182598/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)" \
    || source /home/au182598/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

cd /home/au182598/bullying-project || exit 1

echo "=============================================="
echo "Plan 2 Phase 6: Multimodal Evaluation"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

pip install -q -r plan2_requirements.txt

python plan2_eval/run_full_evaluation.py --config plan2_configs/eval.yaml "$@"

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
if [ -f plan2_eval/full_evaluation_results.json ]; then
    echo "Results summary:"
    python -m json.tool plan2_eval/full_evaluation_results.json | head -60
fi

echo "Phase 6 complete: $(date)"
exit 0
