#!/bin/bash
# Submit the full chained adaptive pipeline with SLURM dependency chaining.
#
# Pipeline:
#   00: Data prep (split + CoT distill + fixed pairs)
#   01: SFT (from base)
#   02: CoT-SFT (resumes from SFT)
#   03: Eval CoT-SFT on train questions (Phase 3.5, single job -- 568 questions)
#   04: Extract adaptive pairs (hard_mining strategy, CPU only)
#   05: ADPO (resumes from CoT-SFT, uses adaptive pairs)
#   06: Final eval (4 parallel parts, ~3050 questions each)
#
# Usage:
#   bash train_model/experiments/20pct_adaptive_v1/sbatch/submit_all.sh

set -e
cd "$(dirname "$0")/../../../.."

DIR="train_model/experiments/20pct_adaptive_v1/sbatch"
EVAL_PARTS=4
mkdir -p train_model/experiments/20pct_adaptive_v1/logs

# Phase 0: Data prep (split + CoT distillation + fixed pair extraction)
DATA_JOB=$(sbatch --parsable "$DIR/00_data_prep.sbatch")
echo "Submitted data prep: $DATA_JOB"

# Phase 1: SFT (depends on data prep)
SFT_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB "$DIR/01_sft.sbatch")
echo "Submitted SFT: $SFT_JOB (depends on $DATA_JOB)"

# Phase 3: CoT-SFT (depends on SFT -- chained, not standalone)
COT_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB "$DIR/02_cot.sbatch")
echo "Submitted CoT-SFT: $COT_JOB (depends on $SFT_JOB)"

# Phase 3.5: Eval CoT-SFT on training questions (568 questions, no split needed)
EVTRAIN_JOB=$(sbatch --parsable --dependency=afterok:$COT_JOB "$DIR/03_eval_train.sbatch")
echo "Submitted eval-on-train: $EVTRAIN_JOB (depends on $COT_JOB)"

# Phase 4: Extract adaptive pairs (depends on eval-on-train, CPU only)
PAIRS_JOB=$(sbatch --parsable --dependency=afterok:$EVTRAIN_JOB "$DIR/04_extract_adaptive.sbatch")
echo "Submitted adaptive pair extraction: $PAIRS_JOB (depends on $EVTRAIN_JOB)"

# Phase 5: ADPO (depends on CoT-SFT adapter + adaptive pairs)
ADPO_JOB=$(sbatch --parsable --dependency=afterok:$PAIRS_JOB "$DIR/05_adpo.sbatch")
echo "Submitted ADPO: $ADPO_JOB (depends on $PAIRS_JOB)"

# Phase 6: Final eval -- split into parallel parts
EVAL_DEPS=""
for p in $(seq 1 $EVAL_PARTS); do
    EVAL_JOB=$(sbatch --parsable --dependency=afterok:$ADPO_JOB \
        --job-name="20pct_av1_eval_p${p}" \
        --output="train_model/experiments/20pct_adaptive_v1/logs/eval_p${p}_%j.out" \
        --error="train_model/experiments/20pct_adaptive_v1/logs/eval_p${p}_%j.err" \
        "$DIR/06_eval.sbatch" --part $p --total-parts $EVAL_PARTS)
    echo "Submitted final eval part $p/$EVAL_PARTS: $EVAL_JOB (depends on $ADPO_JOB)"
    if [ -z "$EVAL_DEPS" ]; then
        EVAL_DEPS="$EVAL_JOB"
    else
        EVAL_DEPS="$EVAL_DEPS:$EVAL_JOB"
    fi
done

echo ""
echo "Pipeline submitted (chained, adaptive hard_mining):"
echo "  00 Data prep:       $DATA_JOB"
echo "  01 SFT:             $SFT_JOB"
echo "  02 CoT-SFT:         $COT_JOB"
echo "  03 Eval-on-train:   $EVTRAIN_JOB"
echo "  04 Adaptive pairs:  $PAIRS_JOB"
echo "  05 ADPO:            $ADPO_JOB"
echo "  06 Final eval:      $EVAL_PARTS parallel parts ($EVAL_DEPS)"
echo ""
echo "Monitor: squeue -u \$USER"
