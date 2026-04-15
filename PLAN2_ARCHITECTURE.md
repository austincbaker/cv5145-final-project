# Plan 2: ADPO + CoT — Full Pipeline Architecture

## Overview

Plan 2 implements a 6-phase pipeline to fine-tune InternVL2.5-8B with Chain-of-Thought reasoning and Anchored Direct Preference Optimization.

**Timeline:** 10 weeks  
**Compute:** 4x A100 80GB (~100-170 GPU-hours total)  
**Base Model:** InternVL2.5-8B (8B parameters)  

## Phase Flow

```
Phase 1: SFT Baseline
    ↓
Phase 2: CoT Data Generation
    ↓
Phase 3: CoT SFT ← (merge datasets)
    ↓
Phase 4: ADPO Preference Pairs
    ↓
Phase 5: ADPO Training ← (init from Phase 3 checkpoint)
    ↓
Phase 6: Evaluation & Ablations
```

## Phase Details

### Phase 1: SFT Baseline (Weeks 1-3)
**Goal:** Establish baseline by fine-tuning on direct answers.

**Inputs:**
- `annotations.json` (2,688 videos)
- Existing question generator with improved distractors

**Outputs:**
- `plan2_data/sft_{train,val,test}.json` (stratified split)
- `plan2_models/sft_baseline/` (checkpoint, used in Phase 3)
- `plan2_eval/sft_baseline_results.json` (baseline accuracy)

**Key Steps:**
1. Regenerate benchmark with improved distractors
   - `compound_aggressor_victim`: 7/7 in-cast
   - `compound_aggressor_action_victim`: 6/7 in-cast
   - `sequence_verification`: 7/7 in-cast
2. Format into SFT triplets (video context + question → answer)
3. Train InternVL2.5-8B + LoRA (rank 8) on 80% of data
4. Evaluate on held-out test set

**Compute:** ~24-48 GPU-hours

---

### Phase 2: CoT Data Generation (Weeks 3-5)
**Goal:** Generate reasoning chains for compound/complex questions via teacher model.

**Inputs:**
- `plan2_data/sft_train.json` (training examples)
- GPT-4o API access (teacher model)
- Annotations (video context)

**Outputs:**
- `plan2_cot/cot_chains_{train,val,test}.json` (reasoning traces)
- `plan2_cot/cot_quality_metrics.json` (filtering statistics)

**Key Steps:**
1. Design CoT prompt template:
   ```
   Video context: [aggressor, victim, action, location, bystanders]
   Question: [Q]
   Correct answer: [A]
   
   Reasoning: Step 1 identify people, Step 2 describe actions, Step 3 assign roles, Step 4 answer
   ```
2. For each compound/complex training question (~4K-6K examples):
   - Call GPT-4o with prompt
   - Extract reasoning chain
   - Validate chain arrives at correct answer
   - Filter out chains referencing unavailable details
3. Keep simple questions (single-role, single-action) with direct answers (no CoT overhead)

**Compute:** ~50 GPU-hours equivalent (API calls, not GPU-bound)

---

### Phase 3: CoT SFT (Weeks 5-7)
**Goal:** Fine-tune on mixed dataset (CoT for compounds, direct for simple).

**Inputs:**
- `plan2_data/sft_{train,val,test}.json` (direct answers)
- `plan2_cot/cot_chains_{train,val,test}.json` (reasoning chains)
- `plan2_models/sft_baseline/` (checkpoint to resume from)

**Outputs:**
- `plan2_models/cot_sft/` (checkpoint, used in Phase 5 ADPO)
- `plan2_eval/cot_sft_results.json` (per-question-type accuracy)

**Key Steps:**
1. Merge datasets:
   - Simple questions: use direct answer format
   - Compound/complex: use CoT format (reasoning → answer)
2. Fine-tune Phase 1 checkpoint on merged data
3. Evaluate:
   - Overall accuracy improvement on compound/complex questions
   - Verify simple question accuracy doesn't degrade

**Training Config:**
- Resume from Phase 1 checkpoint
- Learning rate: 2e-4 (lower, fine-tuning)
- Epochs: 2-3
- Early stopping on val accuracy

**Compute:** ~24-48 GPU-hours

---

### Phase 4: ADPO Preference Pairs (Weeks 7-8)
**Goal:** Extract preference pairs from improved distractor pipeline.

**Inputs:**
- `plan2_data/generated_questions_plan2.json` (questions with hard negatives)
- `plan2_cot/cot_chains_{train}.json` (reasoning for chosen responses)

**Outputs:**
- `plan2_adpo/preference_pairs_{train,val}.json` (40K-60K pairs)
- `plan2_adpo/pair_statistics.json` (hardness distribution)

**Preference Pair Format:**
```json
{
  "video_name": "punch_facebook_001.mp4",
  "question_type": "compound_aggressor_victim",
  "prompt": "Who was the aggressor and who was the victim?",
  "prompt_context": "Aggressor: ...",
  "chosen": "Aggressor: X; Victim: Y",
  "chosen_reasoning": "[reasoning chain from Phase 2, if compound]",
  "rejected": [
    {
      "answer": "Aggressor: Y; Victim: X",
      "hardness": "role_reversal",
      "reason": "roles swapped"
    },
    {
      "answer": "Aggressor: Z; Victim: Y",
      "hardness": "bystander_substitution",
      "reason": "bystander as aggressor"
    },
    ...
  ]
}
```

**Distractor Priority (hard negatives first):**
1. **Role reversals** (aggressor ↔ victim swapped) — ~1,949 → 6,200 (3.2x from Phase 1)
2. **Bystander substitutions** (bystander in role of agg/vic) — ~800 → 2,000
3. **Wrong action, correct roles** (different action, same people) — ~500 → 2,400
4. **Correct action, wrong location** (action visible, different environment) — ~300 → 600

**Filtering:**
- Exclude trivially distinguishable cross-video pairs (completely unrelated contexts)
- 3-5 rejected per chosen for diversity
- **Target:** 40K-60K pairs (revised from distractor quality analysis in TODO_hard_negatives.md)

**Compute:** CPU-bound, no GPU needed

---

### Phase 5: ADPO Training (Weeks 8-10)
**Goal:** Train with ADPO loss to optimize preference ranking.

**Inputs:**
- `plan2_models/cot_sft/` (reference checkpoint)
- `plan2_adpo/preference_pairs_train.json` (40K-60K pairs)

**Outputs:**
- `plan2_models/adpo_final/` (best checkpoint)
- `plan2_eval/adpo_training_metrics.json` (loss curves, accuracy, preference ranking)

**ADPO Loss:**
```
L_ADPO = L_preference(chosen > rejected) + α * L_anchor(model | reference)
```

Where:
- `L_preference`: Bradley-Terry ranking loss (chosen > rejected)
- `L_anchor`: KL divergence from reference (Phase 3 checkpoint)
- `α` (anchoring strength): sweep 0.1 → 1.0

**Training Config:**
- Initialize from Phase 3 checkpoint (reference frozen)
- Batch size: 8 pairs
- Learning rate: 1e-4
- Epochs: 2-3
- Monitor:
  - Val accuracy (test set)
  - Preference accuracy (model scores chosen > rejected)
  - KL divergence from reference
- Hyperparameter sweep:
  - α ∈ [0.1, 0.3, 0.5, 0.7, 1.0]
  - If anchoring too strong (no improvement): reduce α
  - If training unstable: increase α
- Early stopping on val accuracy (primary metric)

**Compute:** ~36-72 GPU-hours

---

### Phase 6: Evaluation & Ablations (Weeks 10-11)
**Goal:** Comprehensive evaluation and ablation study.

**Inputs:**
- `plan2_models/sft_baseline/` (Phase 1)
- `plan2_models/cot_sft/` (Phase 3)
- `plan2_models/adpo_final/` (Phase 5)
- `plan2_data/sft_test.json` (held-out test set)

**Outputs:**
- `plan2_eval/full_comparison.json` (all baselines + ablations)
- `plan2_eval/analysis.md` (written report)
- Figures: accuracy by question type, CoT vs direct, ADPO sensitivity

**Evaluation Matrix:**
1. **Baseline progression:**
   - Zero-shot (no fine-tuning, reported from prior runs)
   - SFT only (Phase 1)
   - SFT + CoT (Phase 3)
   - SFT + CoT + ADPO (Phase 5)

2. **Per-question-type breakdown:**
   - Simple (action, aggressor, victim)
   - Compound (aggressor-location, action-victim, aggressor-victim)
   - Complex (aggressor-action-victim, sequence, role-id)
   - Trick questions

3. **ADPO ablations:**
   - ADPO with vs without CoT pretraining
   - Anchoring strength sensitivity (α sweep results)
   - Preference ranking accuracy (% chosen > rejected)

4. **Qualitative analysis:**
   - Examples where CoT helps (multi-step questions)
   - Examples where CoT doesn't help (simple questions)
   - Failure modes: when reasoning chain is wrong
   - ADPO wins: where preference optimization improves over SFT

5. **Inference metrics:**
   - Tokens generated (SFT direct vs CoT)
   - Wall-clock time per question
   - GPU memory usage

**Compute:** CPU-bound (inference + analysis)

---

## File Structure

```
plan2_sft/
  ├── regenerate_benchmark.py      (Phase 1, step 1)
  ├── format_sft_data.py            (Phase 1, step 2)
  ├── train_sft.py                  (Phase 1, step 3)
  ├── evaluate_sft.py               (Phase 1, step 4)
  ├── run_phase1.sh                 (Phase 1 orchestrator)
  └── README.md

plan2_cot/
  ├── generate_cot_chains.py        (Phase 2)
  ├── quality_filter_cot.py         (Phase 2)
  ├── format_cot_data.py            (Phase 3)
  ├── train_cot_sft.py              (Phase 3)
  ├── evaluate_cot_sft.py           (Phase 3)
  └── README.md

plan2_adpo/
  ├── extract_preference_pairs.py   (Phase 4)
  ├── train_adpo.py                 (Phase 5)
  ├── evaluate_adpo.py              (Phase 5)
  └── README.md

plan2_eval/
  ├── run_full_evaluation.py        (Phase 6)
  ├── generate_report.py            (Phase 6)
  ├── plot_results.py               (Phase 6)
  └── README.md

plan2_data/
  ├── generated_questions_plan2.json
  ├── sft_train.json
  ├── sft_val.json
  ├── sft_test.json
  ├── cot_chains_train.json
  ├── preference_pairs_train.json
  └── [other intermediate files]

plan2_models/
  ├── sft_baseline/
  ├── cot_sft/
  ├── adpo_final/
  └── [checkpoints and logs]
```

## Key Dependencies

- **Phase 1 → Phase 3:** Phase 1 checkpoint (`sft_baseline`) is the reference for Phase 3
- **Phase 1 → Phase 4:** Phase 1 generates the improved questions with hard negatives
- **Phase 2 → Phase 3:** CoT chains are merged with direct answers
- **Phase 2 → Phase 4:** CoT chains are attached to preference pairs (as "chosen_reasoning")
- **Phase 3 → Phase 5:** Phase 3 checkpoint serves as reference model for ADPO anchoring
- **All → Phase 6:** All checkpoints are evaluated in final ablations

## Hyperparameter Summary

| Phase | Model | LR | Epochs | Batch | Key Param |
|-------|-------|-----|--------|-------|-----------|
| 1 | SFT | 5e-4 | 3 | 4 | LoRA rank=8 |
| 3 | CoT SFT | 2e-4 | 2-3 | 4 | Resume Phase 1 |
| 5 | ADPO | 1e-4 | 2-3 | 8 | α sweep 0.1-1.0 |

## Compute Budget

| Phase | Time | Compute |
|-------|------|---------|
| 1 (SFT) | 24-48 hrs | 4x A100 |
| 2 (CoT gen) | 1-2 hrs | API (not GPU) |
| 3 (CoT SFT) | 24-48 hrs | 4x A100 |
| 4 (Pairs) | <1 hr | CPU |
| 5 (ADPO) | 36-72 hrs | 4x A100 |
| 6 (Eval) | 2-4 hrs | CPU + 1x A100 inference |
| **Total** | **~100-170 hrs** | **Mostly 4x A100** |

---

## Running the Full Pipeline

```bash
# Phase 1
chmod +x plan2_sft/run_phase1.sh
./plan2_sft/run_phase1.sh

# Phase 2 & 3
chmod +x plan2_cot/run_phase2_3.sh
./plan2_cot/run_phase2_3.sh

# Phase 4 & 5
chmod +x plan2_adpo/run_phase4_5.sh
./plan2_adpo/run_phase4_5.sh

# Phase 6
python plan2_eval/run_full_evaluation.py
```

Or run individual scripts for each phase as documented in their respective README files.

---

## Success Criteria

**Phase 1:** Baseline accuracy ≥ 30% on test set  
**Phase 3:** CoT improves compound/complex by ≥ 5% abs., simple questions ≥ 0% (no regression)  
**Phase 5:** ADPO improves overall by ≥ 3% abs. from Phase 3  
**Phase 6:** Clear ablation showing contribution of each component (CoT: +5-10%, ADPO: +3-5%)
