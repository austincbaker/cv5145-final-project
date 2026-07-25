#!/bin/bash
# Submit the curriculum pipeline (reuses SFT/CoT from 20pct_adaptive_v1).
#
# Pipeline:
#   04: Extract curriculum pairs (easy for wrong, hard for correct)
#   05: ADPO (resumes from shared CoT-SFT checkpoint)
#   06: Final eval -- ADPO only (4 parallel parts)
#
# Usage:
#   bash train_model/experiments/20pct_curriculum_v1/sbatch/submit_all.sh

set -e
cd "$(dirname "$0")/../../../.."

DIR="train_model/experiments/20pct_curriculum_v1/sbatch"
EVAL_PARTS=4
mkdir -p train_model/experiments/20pct_curriculum_v1/logs

# Phase 4: Extract curriculum pairs (CPU only)
PAIRS_JOB=$(sbatch --parsable "$DIR/04_extract_curriculum.sbatch")
echo "Submitted curriculum pair extraction: $PAIRS_JOB"

# Phase 5: ADPO (depends on pair extraction)
ADPO_JOB=$(sbatch --parsable --dependency=afterok:$PAIRS_JOB "$DIR/05_adpo.sbatch")
echo "Submitted ADPO (curriculum): $ADPO_JOB (depends on $PAIRS_JOB)"

# Phase 6: Final eval -- ADPO only, split into parallel parts
EVAL_DEPS=""
for p in $(seq 1 $EVAL_PARTS); do
    EVAL_JOB=$(sbatch --parsable --dependency=afterok:$ADPO_JOB         --job-name="20pct_cur_eval_p${p}"         --output="train_model/experiments/20pct_curriculum_v1/logs/eval_p${p}_%j.out"         --error="train_model/experiments/20pct_curriculum_v1/logs/eval_p${p}_%j.err"         "$DIR/06_eval.sbatch" --part $p --total-parts $EVAL_PARTS)
    echo "Submitted eval part $p/$EVAL_PARTS: $EVAL_JOB (depends on $ADPO_JOB)"
    if [ -z "$EVAL_DEPS" ]; then
        EVAL_DEPS="$EVAL_JOB"
    else
        EVAL_DEPS="$EVAL_DEPS:$EVAL_JOB"
    fi
done

echo ""
echo "Curriculum pipeline submitted:"
echo "  04 Curriculum pairs: $PAIRS_JOB"
echo "  05 ADPO:             $ADPO_JOB"
echo "  06 Eval:             $EVAL_PARTS parallel parts ($EVAL_DEPS)"
echo ""
echo "Monitor: squeue -u \$USER"
