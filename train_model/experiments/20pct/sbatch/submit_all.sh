#!/bin/bash
# Submit 20% standalone baseline experiments.
# All 4 methods (SFT, CoT, DPO, ADPO) train independently from base.
#
# Usage:
#   bash train_model/experiments/20pct/sbatch/submit_all.sh

set -e
cd "$(dirname "$0")/../../../.."

DIR="train_model/experiments/20pct/sbatch"
EVAL_PARTS=4
mkdir -p train_model/experiments/20pct/logs

# 4 independent training runs (no dependencies between them)
SFT_JOB=$(sbatch --parsable "$DIR/01_sft.sbatch")
echo "Submitted SFT: $SFT_JOB"

COT_JOB=$(sbatch --parsable "$DIR/02_cot.sbatch")
echo "Submitted CoT: $COT_JOB"

DPO_JOB=$(sbatch --parsable "$DIR/03_dpo.sbatch")
echo "Submitted DPO: $DPO_JOB"

ADPO_JOB=$(sbatch --parsable "$DIR/04_adpo.sbatch")
echo "Submitted ADPO: $ADPO_JOB"

# Eval -- depends on all 4 training runs, split into parallel parts
EVAL_DEPS=""
for p in $(seq 1 $EVAL_PARTS); do
    EVAL_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB:$COT_JOB:$DPO_JOB:$ADPO_JOB \
        --job-name="20pct_eval_p${p}" \
        --output="train_model/experiments/20pct/logs/eval_p${p}_%j.out" \
        --error="train_model/experiments/20pct/logs/eval_p${p}_%j.err" \
        "$DIR/05_eval.sbatch" --part $p --total-parts $EVAL_PARTS)
    echo "Submitted eval part $p/$EVAL_PARTS: $EVAL_JOB (depends on all training)"
    if [ -z "$EVAL_DEPS" ]; then
        EVAL_DEPS="$EVAL_JOB"
    else
        EVAL_DEPS="$EVAL_DEPS:$EVAL_JOB"
    fi
done

echo ""
echo "20% standalone pipeline submitted:"
echo "  SFT:  $SFT_JOB"
echo "  CoT:  $COT_JOB"
echo "  DPO:  $DPO_JOB"
echo "  ADPO: $ADPO_JOB"
echo "  Eval: $EVAL_PARTS parallel parts ($EVAL_DEPS)"
echo ""
echo "All training runs are independent (no chaining)."
echo "Monitor: squeue -u \$USER"
