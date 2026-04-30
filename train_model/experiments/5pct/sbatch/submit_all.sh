#!/bin/bash
# Submit all 5% baseline experiments with SLURM dependency chaining.
#
# Usage:
#   bash train_model/experiments/5pct/sbatch/submit_all.sh

set -e
cd "$(dirname "$0")/../../../.."

DIR="train_model/experiments/5pct/sbatch"

# Step 0: Data prep (format split + CoT distillation + pair extraction)
DATA_JOB=$(sbatch --parsable "$DIR/00_data_prep.sbatch")
echo "Submitted data prep: $DATA_JOB"

# Steps 1-4: Four independent training runs, all depend on data prep
SFT_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB "$DIR/01_sft.sbatch")
echo "Submitted SFT: $SFT_JOB (depends on $DATA_JOB)"

COT_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB "$DIR/02_cot.sbatch")
echo "Submitted CoT: $COT_JOB (depends on $DATA_JOB)"

DPO_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB "$DIR/03_dpo.sbatch")
echo "Submitted DPO: $DPO_JOB (depends on $DATA_JOB)"

ADPO_JOB=$(sbatch --parsable --dependency=afterok:$DATA_JOB "$DIR/04_adpo.sbatch")
echo "Submitted ADPO: $ADPO_JOB (depends on $DATA_JOB)"

# Step 5: Eval — depends on all 4 training runs completing
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB:$COT_JOB:$DPO_JOB:$ADPO_JOB "$DIR/05_eval.sbatch")
echo "Submitted eval: $EVAL_JOB (depends on $SFT_JOB, $COT_JOB, $DPO_JOB, $ADPO_JOB)"

echo ""
echo "Pipeline submitted:"
echo "  Data prep:  $DATA_JOB"
echo "  SFT:        $SFT_JOB"
echo "  CoT:        $COT_JOB"
echo "  DPO:        $DPO_JOB"
echo "  ADPO:       $ADPO_JOB"
echo "  Eval:       $EVAL_JOB"
echo ""
echo "Monitor: squeue -u \$USER"
