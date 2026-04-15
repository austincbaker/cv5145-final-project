# Plan 2 Pipeline: Full Execution Guide

Complete 6-phase ADPO + CoT training pipeline for InternVL2.5-8B on video aggression detection.

**Total compute:** ~100-170 GPU-hours on A100, ~10 weeks wall-clock time
**Model:** OpenGVLab/InternVL2_5-8B (8B parameters, vision-language)
**Base dataset:** 2,688 aggressive interaction videos with 6-7/7 in-cast hard negatives

## Pipeline Overview

```
Phase 1: SFT Baseline (Weeks 1-3)
  ↓ [regenerate benchmark + format data + train LoRA + evaluate]
  ↓
Phase 2: CoT Generation (Weeks 3-5)
  ↓ [call GPT-4o for reasoning chains on compound questions]
  ↓
Phase 3: CoT SFT (Weeks 5-7)
  ↓ [resume Phase 1, fine-tune with mixed CoT/direct format]
  ↓
Phase 4: ADPO Pairs (Weeks 7-8)
  ↓ [extract preference pairs with hardness ranking]
  ↓
Phase 5: ADPO Training (Weeks 8-10)
  ↓ [Bradley-Terry + KL divergence anchoring]
  ↓
Phase 6: Evaluation (Weeks 10-11)
  └─ [ablation study: SFT → +CoT → +ADPO]
```

## Running the Pipeline

### Prerequisites
```bash
# Set API key for Phase 2 (GPT-4o calls)
export OPENAI_API_KEY="sk-..."

# Verify access to cluster
ssh au182598@crcv.eecs.ucf.edu

# Pull latest code
cd /home/au182598/bullying-project
git pull origin updating_question_generation
```

### Execution Options

#### Option A: Sequential (Safest)
Run each phase to completion before starting the next.

```bash
# Phase 1: ~24 hours
sbatch plan2_phase1.sbatch
# Monitor: watch squeue -u au182598
# Wait for job to complete

# Phase 2: ~6-12 hours (depends on OpenAI API)
sbatch plan2_phase2.sbatch

# Phase 3: ~20 hours
sbatch plan2_phase3.sbatch

# Phase 4: ~2 hours
sbatch plan2_phase4.sbatch

# Phase 5: ~20 hours
sbatch plan2_phase5.sbatch

# Phase 6: ~4 hours
sbatch plan2_phase6.sbatch
```

#### Option B: Parallel (Fast-tracked)
Start Phase 2 while Phase 1 is training. Phase 2 and 3 don't depend on GPU (Phase 2 uses CPU+API, Phase 3 needs Phase 1 output but can wait).

```bash
# Start Phase 1
sbatch plan2_phase1.sbatch  # JOB_ID_1

# Immediately start Phase 2 (no GPU dependency)
sbatch plan2_phase2.sbatch  # JOB_ID_2

# Monitor both
watch 'squeue -u au182598 | grep plan2'

# Once Phase 1 completes and Phase 2 finishes, chain the rest
```

### Monitoring Progress

```bash
# Check job status
squeue -u au182598

# View job output
tail -f plan2_phase1_<jobid>.out

# Check GPU usage
ssh au182598@crcv.eecs.ucf.edu "nvidia-smi"

# Count output files
ls -la plan2_models/
ls -la plan2_data/
ls -la plan2_cot/
ls -la plan2_adpo/
ls -la plan2_eval/
```

## Phase Details

### Phase 1: SFT Baseline (plan2_phase1.sbatch)
- **Input:** generated_questions_20260327_201047.json
- **Process:**
  1. Regenerate questions with improved hard negatives
  2. Format into SFT training data (80/10/10 split)
  3. Train InternVL2.5-8B with LoRA (rank 8, α 16)
  4. Evaluate on test set
- **Output:** 
  - `plan2_models/sft_baseline/` (LoRA adapter)
  - `plan2_data/sft_{train,val,test}.json` (formatted data)
  - `plan2_eval/sft_baseline_results.json` (metrics)
- **Hyperparameters:**
  - Learning rate: 5e-4
  - Epochs: 3
  - Batch size: 4 (per GPU)
  - Mixed precision: BF16
  - Gradient accumulation: 2
  - Max grad norm: 1.0

### Phase 2: CoT Generation (plan2_phase2.sbatch)
- **Input:** `plan2_data/sft_train.json`
- **Process:**
  - Identify CoT-eligible types (compounds, complex questions)
  - Call GPT-4o with structured prompt for reasoning chains
  - Quality filter (chains must mention key answer tokens)
  - Merge chains into training examples
- **Output:** `plan2_cot/cot_chains_train.json`
  - Fields: `reasoning_chain` (str), `used_cot` (bool)
  - ~60-70% of compound questions will have CoT chains
- **Cost:** ~$50-150 (depends on sample rate)
- **Dependencies:**
  - OPENAI_API_KEY environment variable
  - Can run in parallel with Phase 1

### Phase 3: CoT SFT (plan2_phase3.sbatch)
- **Input:** 
  - Phase 1 checkpoint: `plan2_models/sft_baseline/`
  - CoT chains: `plan2_cot/cot_chains_train.json`
  - Validation data: `plan2_data/sft_val.json`
- **Process:**
  - Resume from Phase 1 LoRA weights
  - Mixed format training:
    - Compounds with CoT: "Reasoning:\n{chain}\n\nAnswer: {answer}"
    - Simple questions: "Answer: {answer}"
  - Lower learning rate (2e-4) to preserve Phase 1 knowledge
- **Output:** `plan2_models/cot_sft/` (LoRA adapter)
- **Hyperparameters:**
  - Learning rate: 2e-4
  - Epochs: 2
  - Batch size: 4
  - Gradient accumulation: 2
  - Early stopping on eval loss
- **Dependencies:** Phase 1 + Phase 2

### Phase 4: ADPO Preference Pairs (plan2_phase4.sbatch)
- **Input:**
  - Questions: `plan2_data/generated_questions_plan2.json`
  - CoT chains (optional): `plan2_cot/cot_chains_train.json`
- **Process:**
  - Extract preference pairs from distractor pipeline
  - Chosen: correct_answer + optional reasoning_chain
  - Rejected: hard negatives ranked by hardness:
    1. Role reversals (3.2x improvement expected)
    2. Bystander substitutions
    3. Wrong action, correct roles
    4. Cross-video distractors (fallback)
  - 3 rejected per chosen pair (configurable)
- **Output:** `plan2_adpo/preference_pairs_train.json`
  - Fields per pair: `prompt`, `chosen`, `rejected` (list with hardness)
  - ~2,500-3,000 pairs
- **Dependencies:** Phase 1 (for questions), Phase 2 (optional, for CoT chains)

### Phase 5: ADPO Training (plan2_phase5.sbatch)
- **Input:**
  - Phase 3 checkpoint: `plan2_models/cot_sft/`
  - Preference pairs: `plan2_adpo/preference_pairs_train.json`
- **Process:**
  - ADPO loss: L = L_pref + α * L_kl
    - L_pref: Bradley-Terry ranking loss (chosen > rejected)
    - L_kl: KL divergence from Phase 3 (anchoring)
  - Resume from Phase 3 checkpoint
  - Frozen reference model (Phase 3) for KL computation
  - α default: 0.5 (can sweep 0.1-1.0 for sensitivity)
- **Output:** `plan2_models/adpo_final/` (LoRA adapter)
- **Hyperparameters:**
  - Learning rate: 1e-4
  - Epochs: 2
  - Batch size: 8
  - Anchoring strength α: 0.5
  - Max grad norm: 1.0
- **Dependencies:** Phase 3 + Phase 4

### Phase 6: Evaluation (plan2_phase6.sbatch)
- **Input:**
  - Phase 1 checkpoint: `plan2_models/sft_baseline/`
  - Phase 3 checkpoint: `plan2_models/cot_sft/`
  - Phase 5 checkpoint: `plan2_models/adpo_final/`
  - Test data: `plan2_data/sft_test.json`
- **Process:**
  - Evaluate each stage on test set
  - Metrics:
    - Overall accuracy
    - Per-question-type breakdown
    - Trick vs normal question accuracy
    - Ablation comparison (SFT → +CoT → +ADPO)
- **Output:** `plan2_eval/full_evaluation_results.json`
  - Full metrics table for paper
- **Dependencies:** All previous phases

## Troubleshooting

### Phase 1 fails with "AssocMaxGRESPerJob"
```
GPU/memory request exceeds cluster limits. Modify plan2_phase1.sbatch:
--gres=gpu:ampere:1  (was 4, reduced to 1)
--mem=32G            (was 256G, reduced to 32G)
```
Current config (1x A100, 32GB) is accepted.

### Phase 1 fails with "untracked working tree files"
```bash
# Clean untracked files before running
git clean -fd
git status  # Should show clean slate
```

### Phase 2 fails with OPENAI_API_KEY error
```bash
# Set API key in current session
export OPENAI_API_KEY="sk-your-key-here"

# Or add to sbatch script (not recommended for security)
# Safer: set in cluster environment
ssh au182598@crcv.eecs.ucf.edu
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

### Phase 2 is slow (OpenAI rate limits)
```bash
# Reduce sample rate to save cost and time
python plan2_cot/generate_cot_chains.py --sample-rate 0.5
# Then rerun Phase 3/4/5 on reduced dataset
```

### Phase 3+ fails with "checkpoint not found"
```
Ensure previous phase completed successfully:
ls -la plan2_models/sft_baseline/adapter_config.json
ls -la plan2_cot/cot_chains_train.json
```
If missing, rerun the previous phase or copy from backup.

### GPU runs out of memory
Reduce batch size in sbatch scripts:
```bash
--batch-size 4  # Phase 1, 3
--batch-size 8  # Phase 5
# Reduce further if needed
```

## Output Files Checklist

After complete pipeline:
```
plan2_models/
  sft_baseline/          ✓ Phase 1
  cot_sft/               ✓ Phase 3
  adpo_final/            ✓ Phase 5

plan2_data/
  generated_questions_plan2.json    ✓ Phase 1
  sft_train.json                    ✓ Phase 1
  sft_val.json                      ✓ Phase 1
  sft_test.json                     ✓ Phase 1

plan2_cot/
  cot_chains_train.json             ✓ Phase 2

plan2_adpo/
  preference_pairs_train.json       ✓ Phase 4

plan2_eval/
  full_evaluation_results.json      ✓ Phase 6
  sft_baseline_results.json         ✓ Phase 1
```

## Timeline Expectations

- **Phase 1:** 20-24 hours (regenerate + format + 3 epochs + eval)
- **Phase 2:** 6-12 hours (depends on OpenAI API throughput)
- **Phase 3:** 18-24 hours (2 epochs on 4K+ examples)
- **Phase 4:** 1-2 hours (pure Python, no GPU)
- **Phase 5:** 18-24 hours (2 epochs preference training)
- **Phase 6:** 4-6 hours (inference on 3 models)

**Total wall-clock:** ~100-150 hours if run sequentially (~4-6 weeks)

**With parallelization (Phase 2 during Phase 1):** ~90-140 hours (~4-6 weeks, minimal savings due to Phase 2 length)

## Next Steps

1. **Commit all Phase files:**
   ```bash
   git add plan2_*
   git commit -m "Implement Plan 2 phases 2-6 (CoT + ADPO pipeline)"
   git push origin updating_question_generation
   ```

2. **Start Phase 1:**
   ```bash
   ssh au182598@crcv.eecs.ucf.edu "cd /home/au182598/bullying-project && sbatch plan2_phase1.sbatch"
   ```

3. **Monitor progress:**
   ```bash
   watch 'ssh au182598@crcv.eecs.ucf.edu squeue -u au182598 | grep plan2'
   ```

4. **Once Phase 1 completes, start Phase 2:**
   ```bash
   ssh au182598@crcv.eecs.ucf.edu "cd /home/au182598/bullying-project && sbatch plan2_phase2.sbatch"
   ```

5. **Chain remaining phases as each completes.**

## Key Concepts

**ADPO (Anchored Direct Preference Optimization):**
- Combines preference learning (Bradley-Terry) with KL divergence anchoring
- Prevents catastrophic forgetting of Phase 3 knowledge
- α controls anchor strength (sweep 0.1-1.0 for sensitivity)

**Hard Negative Mining:**
- Role reversals (swap aggressor/victim): 3.2x improvement expected
- Bystander substitutions: in-cast people as wrong answer
- Wrong action: different action mentioned but correct roles
- Cross-video: fallback for questions with insufficient distractors

**Chain-of-Thought (CoT):**
- For compound/complex questions only (not all questions)
- Generated via GPT-4o with quality filtering
- Format: "Reasoning:\n{steps}\n\nAnswer: {answer}"
- Expected 5-10% accuracy improvement on compound questions

## References

- PLAN2_ARCHITECTURE.md - Detailed technical specification
- PLAN2_STATUS.md - Current implementation status
- plan2_sft/README.md - Phase 1 documentation
- /home/au182598/bullying-project/plan2_*/ - All pipeline code
