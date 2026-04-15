#!/bin/bash

# Phase 1: SFT Baseline Pipeline
# Run all steps: regenerate → format → train → evaluate

set -e  # Exit on error

echo "================================"
echo "PHASE 1: SFT BASELINE TRAINING"
echo "================================"

# Configuration
ANNOTATIONS="annotations.json"
SEED=42
EPOCHS=3
BATCH_SIZE=4
LR=5e-4

# Step 1: Regenerate questions
echo ""
echo "Step 1: Regenerate benchmark with improved distractors..."
python plan2_sft/regenerate_benchmark.py

# Step 2: Format SFT data
echo ""
echo "Step 2: Format SFT training data..."
python plan2_sft/format_sft_data.py

# Step 3: Train SFT baseline
echo ""
echo "Step 3: Train SFT baseline on InternVL2.5-8B..."
python plan2_sft/train_sft.py \
  --train-data plan2_data/sft_train.json \
  --val-data plan2_data/sft_val.json \
  --output-dir plan2_models/sft_baseline \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR

# Step 4: Evaluate on test set
echo ""
echo "Step 4: Evaluate SFT baseline..."
python plan2_sft/evaluate_sft.py \
  --model plan2_models/sft_baseline \
  --test-data plan2_data/sft_test.json \
  --output plan2_eval/sft_baseline_results.json

echo ""
echo "================================"
echo "PHASE 1 COMPLETE"
echo "================================"
echo "Checkpoint saved to: plan2_models/sft_baseline"
echo "Results saved to: plan2_eval/sft_baseline_results.json"
