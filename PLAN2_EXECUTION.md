# Plan 2 Execution Guide with QOS Constraints

Complete instructions for submitting and running the Plan 2 pipeline within cluster QOS limits.

## QOS Constraints (group3-1)

```
MaxTRESPerJob:  cpu=12, gres/gpu=4    (max 4 GPUs per single job)
MaxTRESPerUser: gres/gpu=4            (max 4 GPUs TOTAL across all user jobs)
MaxWall:        14 days               (max 14 day wall time per job)
```

**Critical constraint:** You can only have 4 GPUs in use at any given time. This means:
- If Phase 1 (using 4 GPUs) is running, no other GPU job can start
- CPU-only jobs (Phase 2, 4) don't consume GPU quota and can run in parallel

## Optimized Execution Strategy

```
Timeline:
┌─────────────────────────────────────────────────────────┐
│ Phase 1: SFT (4 GPU)           [~24 hours]              │
│ Phase 2: CoT (CPU-only) ────→ [~6-12 hours, parallel] │
└─────────────────────────────────────────────────────────┘
         ↓ (after Phase 1 completes)
┌──────────────────┐
│ Phase 3 (1 GPU)  │ [~20 hours]
└──────────────────┘
         ↓
┌──────────────────┐
│ Phase 4 (CPU)    │ [~2 hours]
└──────────────────┘
         ↓
┌──────────────────┐
│ Phase 5 (1 GPU)  │ [~20 hours]
└──────────────────┘
         ↓
┌──────────────────┐
│ Phase 6 (1 GPU)  │ [~4 hours]
└──────────────────┘
```

**Total wall-clock:** ~100-120 hours (~4-5 days sequential)

## Resource Allocation Per Phase

| Phase | Name | Type | GPUs | CPUs | Memory | Wall Time |
|-------|------|------|------|------|--------|-----------|
| 1 | SFT Baseline | Training | 4 | 12 | 180GB | 24h |
| 2 | CoT Generation | API/CPU | 0 | 8 | 32GB | 12h |
| 3 | CoT SFT | Training | 1 | 8 | 48GB | 24h |
| 4 | ADPO Pairs | CPU | 0 | 8 | 32GB | 4h |
| 5 | ADPO Training | Training | 1 | 8 | 48GB | 24h |
| 6 | Evaluation | Inference | 1 | 8 | 48GB | 8h |

**Key points:**
- Phase 1 uses all 4 GPUs for speed (can't parallelize further)
- Phase 2 runs while Phase 1 trains (CPU-only, no GPU conflict)
- Phases 3, 5, 6 use 1 GPU each (can run one at a time due to sequential nature)
- Phase 4 runs CPU-only after Phase 1

## Execution Commands

### Step 1: Prepare and Submit Phase 1

```bash
cd /home/au182598/bullying-project
git pull origin updating_question_generation

# Cancel any old jobs
scancel 337106  # Or previous job ID

# Submit Phase 1 (uses 4 GPUs)
sbatch plan2_phase1.sbatch
# Returns: Submitted batch job <JOB_ID_1>
```

Check status:
```bash
squeue -u au182598  # Monitor Phase 1
```

### Step 2: Start Phase 2 While Phase 1 Trains

Once Phase 1 is confirmed running:
```bash
# Verify Phase 1 is running
squeue -u au182598 | grep plan2_phase1

# Set OpenAI API key
export OPENAI_API_KEY="sk-..."

# Submit Phase 2 (CPU-only, no GPU conflict)
sbatch plan2_phase2.sbatch
# Returns: Submitted batch job <JOB_ID_2>
```

Monitor both:
```bash
watch -n 30 'squeue -u au182598 | grep plan2'
```

### Step 3: Wait for Phase 1 Completion

```bash
# Check Phase 1 output
tail -f plan2_phase1_<JOB_ID_1>.out

# When you see "Phase 1 Complete!", proceed to Step 4
# Verify output files exist
ls -la plan2_data/sft_train.json
ls -la plan2_models/sft_baseline/adapter_config.json
```

### Step 4: After Phase 1 Completes - Chain Remaining Phases

Once Phase 1 finishes, submit phases sequentially:

```bash
# Phase 3 (resume from Phase 1 with CoT chains)
sbatch plan2_phase3.sbatch
# wait for completion

# Phase 4 (extract preference pairs)
sbatch plan2_phase4.sbatch
# wait for completion

# Phase 5 (ADPO training)
sbatch plan2_phase5.sbatch
# wait for completion

# Phase 6 (evaluation)
sbatch plan2_phase6.sbatch
# wait for completion
```

Or create a monitoring script:

```bash
#!/bin/bash
# submit_remaining_phases.sh

check_and_submit() {
    local phase=$1
    local depends_on=$2
    
    while true; do
        # Check if dependency job is done
        if ! squeue -j $depends_on > /dev/null 2>&1; then
            echo "[$phase] Dependency $depends_on complete, submitting..."
            sbatch plan2_phase${phase}.sbatch
            break
        fi
        echo "[$phase] Waiting for job $depends_on..."
        sleep 60
    done
}

# Phase 1 job ID
PHASE1_JOB=$(squeue -u au182598 | grep plan2_phase1 | awk '{print $1}')
echo "Phase 1 Job: $PHASE1_JOB"

# After Phase 1, submit Phase 3
check_and_submit 3 $PHASE1_JOB
PHASE3_JOB=$(squeue -u au182598 | grep plan2_phase3 | awk '{print $1}')

# After Phase 3, submit Phase 4
check_and_submit 4 $PHASE3_JOB
PHASE4_JOB=$(squeue -u au182598 | grep plan2_phase4 | awk '{print $1}')

# And so on...
```

## Monitoring Commands

### Real-time status
```bash
# Watch all Plan 2 jobs
watch -n 30 'squeue -u au182598 | grep plan2'

# Detailed status
squeue -u au182598 -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

### Check job outputs
```bash
# Phase N output
tail -100 plan2_phase<N>_<JOB_ID>.out
tail -100 plan2_phase<N>_<JOB_ID>.err

# Live streaming
tail -f plan2_phase<N>_<JOB_ID>.out
```

### Check resource usage
```bash
# SSH to cluster and check GPU usage
ssh au182598@crcv.eecs.ucf.edu "nvidia-smi"

# Or from here
squeue -u au182598 --Format=jobid,name,tres-per-node,minmemory
```

## Troubleshooting

### Job Rejected: "AssocMaxGRESPerJob"
```
The job requested more GPUs than the QOS limit allows.
Check Phase 1 sbatch: should request --gres=gpu:ampere:4 (maximum for group3-1)
All other GPU phases: should request --gres=gpu:ampere:1
```

### Job Rejected: "QOSMaxGRESPerUser"
```
You already have too many GPUs in use across all jobs.
Wait for running GPU jobs to complete before submitting new GPU jobs.

Check:
  squeue -u au182598
  
Only CPU-only phases (2, 4) can run in parallel with GPU phases.
```

### Job Rejected: "QOSMaxCPUsPerUser"
```
This is not your limit (group3-1 allows 12 CPUs per job, unlimited total).
Likely a system-wide CPU limit. Retry in a few minutes.
```

### Job Runs But Out of Memory
```
Reduce batch size in the training script:
  plan2_sft/train_sft.py:       --batch-size 2  (was 4)
  plan2_cot/train_cot_sft.py:   --batch-size 2  (was 4)
  plan2_adpo/train_adpo.py:     --batch-size 4  (was 8)
```

### Python Import Errors (prompt_generator not found)
```
Verify PYTHONPATH is set in sbatch:
  export PYTHONPATH="/home/au182598/bullying-project:${PYTHONPATH}"

Should be in the "Environment setup" section of each sbatch.
```

### Missing Phase N Output
```
Check if Phase N-1 completed successfully:
  ls -la plan2_data/sft_train.json        (Phase 1 output)
  ls -la plan2_cot/cot_chains_train.json  (Phase 2 output)
  ls -la plan2_models/cot_sft/            (Phase 3 output)
  
If missing, rerun the previous phase.
```

## Expected Output Files

After completing all phases:

```
✓ Phase 1 outputs:
  plan2_models/sft_baseline/              (LoRA adapter)
  plan2_data/sft_train.json               (4K+ training examples)
  plan2_data/sft_val.json                 (validation set)
  plan2_data/sft_test.json                (test set)
  plan2_eval/sft_baseline_results.json    (Phase 1 eval metrics)

✓ Phase 2 outputs:
  plan2_cot/cot_chains_train.json         (CoT chains for compounds)
  
✓ Phase 3 outputs:
  plan2_models/cot_sft/                   (LoRA adapter with CoT)
  
✓ Phase 4 outputs:
  plan2_adpo/preference_pairs_train.json  (ranked preference pairs)
  
✓ Phase 5 outputs:
  plan2_models/adpo_final/                (ADPO-optimized LoRA adapter)
  
✓ Phase 6 outputs:
  plan2_eval/full_evaluation_results.json (ablation study + metrics)
```

## Quick Reference: Copy-Paste Commands

```bash
# Full pipeline submission (manual sequential)
cd /home/au182598/bullying-project
git pull origin updating_question_generation

# Phase 1 + 2 (Phase 2 can run in parallel)
sbatch plan2_phase1.sbatch
sleep 5  # Give Phase 1 time to start
sbatch plan2_phase2.sbatch

# Wait for Phase 1 to complete, then continue
# (Check with: squeue -u au182598)

# Phase 3-6 (submit one at a time, wait for each to complete)
sbatch plan2_phase3.sbatch  # wait ~20h
sbatch plan2_phase4.sbatch  # wait ~2h
sbatch plan2_phase5.sbatch  # wait ~20h
sbatch plan2_phase6.sbatch  # wait ~4h

# Monitor progress
watch -n 30 'squeue -u au182598 | grep plan2'
```

## Notes

1. **Phase 2 timing:** Depends on OpenAI API throughput. May take 6-12 hours. Can be paused/restarted if needed.

2. **GPU efficiency:** Phases 3, 5, 6 each use 1 GPU. Could increase to 2-4 GPUs for speed if willing to wait longer, but would need to run sequentially.

3. **Walltime buffer:** Each phase has 24h+ walltime to account for queueing and compute. Requeue is enabled, so if a job times out, it will automatically resubmit.

4. **Cost estimate:** ~100-120 GPU-hours total (A100 equivalent). Check institutional compute budget.

5. **Checkpoints:** All phases save checkpoints, so jobs can resume on requeue without restarting from scratch.

## Final Validation

Before considering the pipeline complete, verify:

```bash
# Check all model checkpoints exist
find plan2_models -name "adapter_config.json" | wc -l  # Should be 3+

# Check all data files exist
ls -la plan2_data/sft_*.json plan2_adpo/preference_pairs_train.json
ls -la plan2_cot/cot_chains_train.json

# Check evaluation results
cat plan2_eval/full_evaluation_results.json | python -m json.tool | head -30
```

If all checkpoints and data files exist, the pipeline is complete and ready for inference/publication.
