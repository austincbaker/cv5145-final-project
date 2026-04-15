# Plan 2: Quick Start Guide

## 1-Minute Setup

```bash
# Install dependencies
pip install -r plan2_requirements.txt

# Run Phase 1 (SFT baseline)
cd /home/austin/cv-research/bullying-project
chmod +x plan2_sft/run_phase1.sh
./plan2_sft/run_phase1.sh
```

That's it! Phase 1 will:
1. ✅ Regenerate benchmark with improved distractors
2. ✅ Format SFT training data (80/10/10 split)
3. ✅ Train InternVL2.5-8B + LoRA for 3 epochs
4. ✅ Evaluate on test set

**Expected Runtime:** 24-48 hours on 4x A100 GPU

---

## After Phase 1 Completes

Check the results:
```bash
cat plan2_eval/sft_baseline_results.json
```

You should see:
```json
{
  "overall_accuracy": ~30-40%,
  "by_question_type": {
    "compound_aggressor_victim": ~35%,
    "compound_aggressor_action_victim": ~30%,
    ...
  }
}
```

---

## Phase 1 Outputs

After `run_phase1.sh` completes, you'll have:

| File | Purpose |
|------|---------|
| `plan2_data/generated_questions_plan2.json` | Regenerated benchmark (6-7/7 hard negatives) |
| `plan2_data/sft_train.json` | Training set (80%) |
| `plan2_data/sft_val.json` | Validation set (10%) |
| `plan2_data/sft_test.json` | Test set (10%) |
| `plan2_models/sft_baseline/` | **Saved checkpoint** (use for Phase 3) |
| `plan2_eval/sft_baseline_results.json` | Accuracy breakdown |

---

## Customizing Phase 1

Edit `plan2_sft/run_phase1.sh`:

```bash
# Adjust training hyperparameters
EPOCHS=3          # Change to 2 for faster iteration
BATCH_SIZE=4      # Reduce to 2 if GPU OOM
LR=5e-4           # Learning rate
```

Or run individual scripts:
```bash
python plan2_sft/regenerate_benchmark.py
python plan2_sft/format_sft_data.py
python plan2_sft/train_sft.py --epochs 2 --batch-size 2
python plan2_sft/evaluate_sft.py
```

---

## Next: Phase 2 (Weeks 3-5)

Once Phase 1 is done, Phase 2 will:
1. Generate reasoning chains via GPT-4o
2. Merge with training data
3. Fine-tune on mixed CoT + direct answers

**Prerequisite:** Set OpenAI API key
```bash
export OPENAI_API_KEY="sk-..."
```

Phase 2 script (coming soon):
```bash
chmod +x plan2_cot/run_phase2_3.sh
./plan2_cot/run_phase2_3.sh
```

---

## Troubleshooting

**GPU OOM?**
```bash
./plan2_sft/run_phase1.sh --batch-size 2
```

**Model loading fails?**
```bash
huggingface-cli login
# Enter your Hugging Face token
```

**Want to resume training?**
Training automatically saves checkpoints every 500 steps and uses early stopping, so you can safely Ctrl+C and restart.

---

## Architecture Overview

```
Phase 1: SFT Baseline
    ↓ (checkpoint saved)
Phase 2: CoT Data Generation
    ↓ (merge datasets)
Phase 3: CoT SFT Fine-tuning
    ↓ (use as reference)
Phase 5: ADPO Training
    ↓
Phase 6: Final Evaluation
```

See `PLAN2_ARCHITECTURE.md` for full details.

---

## Key Files

- **Phase 1 details:** `plan2_sft/README.md`
- **Full pipeline spec:** `PLAN2_ARCHITECTURE.md`
- **Status & progress:** `PLAN2_STATUS.md`
- **Requirements:** `plan2_requirements.txt`

---

## Questions?

- **Phase 1 details:** See `plan2_sft/README.md`
- **Full pipeline:** See `PLAN2_ARCHITECTURE.md`
- **Debugging:** See troubleshooting section above
