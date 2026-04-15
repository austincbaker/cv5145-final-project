# Plan 2: ADPO + CoT — Implementation Status

**Last Updated:** 2026-04-14  
**Timeline:** 10 weeks (Weeks 1-3 Phase 1 active)  
**Base Model:** InternVL2.5-8B  
**GPU:** 4x A100 80GB  

---

## Phase 1: SFT Baseline — ✅ SCAFFOLDING COMPLETE

### Completed
- [x] Phase 1 directory structure created (`plan2_sft/`)
- [x] Regeneration script (`regenerate_benchmark.py`)
  - Regenerates with improved distractors (6-7/7 in-cast)
  - InternVL2.5-8B compatible
- [x] SFT data formatter (`format_sft_data.py`)
  - 80/10/10 stratified train/val/test split
  - Outputs: `sft_{train,val,test}.json`
- [x] Training script (`train_sft.py`)
  - LoRA fine-tuning (rank 8)
  - Validation every 500 steps, early stopping
  - BF16 mixed precision, gradient accumulation
- [x] Evaluation script (`evaluate_sft.py`)
  - Per-question-type accuracy
  - Trick vs non-trick breakdown
- [x] Pipeline orchestrator (`run_phase1.sh`)
  - Single command runs all 4 steps
- [x] Documentation
  - `plan2_sft/README.md` (Phase 1 details)
  - `PLAN2_ARCHITECTURE.md` (full 6-phase pipeline)
  - `plan2_requirements.txt` (dependencies)

### Ready to Run
```bash
# Install dependencies
pip install -r plan2_requirements.txt

# Run Phase 1
chmod +x plan2_sft/run_phase1.sh
./plan2_sft/run_phase1.sh
```

**Expected Output:**
- `plan2_data/sft_*.json` (training data)
- `plan2_models/sft_baseline/` (checkpoint + LoRA weights)
- `plan2_eval/sft_baseline_results.json` (accuracy metrics)

**Expected Runtime:** 24-48 hours on 4x A100

### Next Steps
1. **Run Phase 1** to establish baseline
2. **Review Phase 1 results** (baseline accuracy by question type)
3. **Move to Phase 2** (CoT data generation via GPT-4o)

---

## Phase 2: CoT Data Generation — ⏳ PENDING

**Status:** Scaffolding in progress  
**Weeks:** 3-5

**To Implement:**
- [ ] CoT prompt engineering
- [ ] GPT-4o API batch calls
- [ ] Quality filtering (chains must reach correct answer)
- [ ] Data merging (CoT for compounds, direct for simple)

---

## Phase 3: CoT SFT — ⏳ PENDING

**Status:** Design complete, implementation awaiting Phase 2  
**Weeks:** 5-7

**Strategy:**
- Resume from Phase 1 checkpoint
- Fine-tune on mixed dataset (CoT + direct answers)
- Measure improvement on compound/complex questions

---

## Phase 4: ADPO Preference Pairs — ⏳ PENDING

**Status:** Design complete  
**Weeks:** 7-8

**Strategy:**
- Extract from improved distractor pipeline (6-7/7 hard negatives)
- Prioritize: role reversals (3.2x gain) → bystander subs → wrong action
- Target: 40K-60K pairs (refined from distractor analysis)

---

## Phase 5: ADPO Training — ⏳ PENDING

**Status:** Design complete  
**Weeks:** 8-10

**Strategy:**
- Initialize from Phase 3 checkpoint
- ADPO loss with anchoring strength α sweep (0.1-1.0)
- Monitor: accuracy, preference ranking, KL divergence
- Early stopping on validation accuracy

---

## Phase 6: Evaluation & Ablations — ⏳ PENDING

**Status:** Design complete  
**Weeks:** 10-11

**Strategy:**
- Ablations: SFT → SFT+CoT → SFT+CoT+ADPO
- Per-question-type analysis
- Hyperparameter sensitivity (α sweep)
- Qualitative reasoning traces

---

## Directory Layout

```
plan2_sft/              ✅ Phase 1 (ready)
  ├── regenerate_benchmark.py
  ├── format_sft_data.py
  ├── train_sft.py
  ├── evaluate_sft.py
  ├── run_phase1.sh
  └── README.md

plan2_cot/              ⏳ Phase 2-3 (pending)
plan2_adpo/             ⏳ Phase 4-5 (pending)
plan2_eval/             ⏳ Phase 6 (pending)

plan2_data/             (will contain train/val/test splits)
plan2_models/           (will contain checkpoints)

PLAN2_ARCHITECTURE.md   ✅ (full pipeline spec)
PLAN2_STATUS.md         ✅ (this file)
plan2_requirements.txt  ✅ (dependencies)
```

---

## Key Metrics to Track

### Phase 1 (SFT Baseline)
- Overall test accuracy: **target ≥ 30%**
- Simple questions accuracy
- Compound questions accuracy
- Trick question accuracy

### Phase 3 (CoT SFT)
- Compound/complex improvement: **target +5% abs.**
- Simple question regression: **target ≥ 0% (no degradation)**

### Phase 5 (ADPO)
- Overall improvement from Phase 3: **target +3% abs.**
- Preference ranking accuracy: **target ≥ 70% (chosen > rejected)**

### Phase 6 (Final)
- Ablation breakdown (contribution of each component)
- Per-question-type win/loss analysis
- Anchoring strength sensitivity plot

---

## Dependencies & Setup

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r plan2_requirements.txt

# Verify InternVL2.5-8B can be loaded
python -c "from transformers import AutoModel; print('OK')"

# (For Phase 2) Set OpenAI API key
export OPENAI_API_KEY="sk-..."
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'prompt_generator'"
- Ensure you're running from the project root: `/home/austin/cv-research/bullying-project/`

### "CUDA OOM" during training
- Reduce `--batch-size` (default 4, try 2 or 1)
- Increase `--lr` slightly to compensate

### "InternVL2.5-8B" model loading fails
- Ensure `trust_remote_code=True` is passed (scripts do this)
- Check Hugging Face credentials: `huggingface-cli login`

---

## Timeline & Milestones

| Week | Phase | Milestone |
|------|-------|-----------|
| 1-3 | Phase 1 | ✅ SFT baseline (accuracy benchmark) |
| 3-5 | Phase 2 | ⏳ CoT chains generated (quality filtered) |
| 5-7 | Phase 3 | ⏳ CoT SFT trained (improvement on compounds) |
| 7-8 | Phase 4 | ⏳ Preference pairs extracted (40K-60K pairs) |
| 8-10 | Phase 5 | ⏳ ADPO final model trained (α sweep complete) |
| 10-11 | Phase 6 | ⏳ Full evaluation & ablations (report written) |

---

## Questions & Next Actions

1. **Ready to start Phase 1?** Run `./plan2_sft/run_phase1.sh`
2. **Need to adjust hyperparameters?** Edit `plan2_sft/run_phase1.sh` or pass flags
3. **GPU budget concern?** Phase 1 is ~24-48 hours; Phases 2-3 are similar; Phase 5 is highest (~72 hrs)

---

See `PLAN2_ARCHITECTURE.md` for detailed specifications and `plan2_sft/README.md` for Phase 1 details.
