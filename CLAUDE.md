# Project: Video Aggression Detection Benchmark

NeurIPS 2026 submission -- benchmark + progressive fine-tuning pipeline for VLM-based video aggression detection.

## Server

- SLURM cluster: `au182598@crcv.eecs.ucf.edu`
- SSH: `sshpass -p "$UCF_SERVER_TOKEN" ssh -o StrictHostKeyChecking=no au182598@crcv.eecs.ucf.edu`
- File transfer: SFTP works. `scp` fails due to password special characters. For large files use `ssh cat` redirect or SFTP.
- Project path on server: `~/aggressive_behavior_project`
- Conda envs: `vlm_py312`, `vlm_py312_fromsrc`, `vlm_py312_tf4451`, `vlm_train_py312`
- SLURM Job Time: `Never set a job time limit when submitting to the cluster.`
- SBATCH default file parameters: `When generating sbatch scripts the following fields should always be set: #SBATCH --partition=gpu #SBATCH --qos=group3 #SBATCH --output=<relevant name>_%j.out #SBATCH --error=<relevant name>_%j.err #SBATCH --requeue #SBATCH --signal=B:USR1@120 #SBATCH --open-mode=append`

## Encoding -- CRITICAL

The cluster runs `LC_ALL=C` (Latin-1). Any non-ASCII character in Python stdout/stderr crashes the job.

- All `print()` output must be ASCII-safe. Use `.encode('ascii', 'replace').decode()` on any string that might contain Unicode (especially model-generated text).
- All `open()` calls for reading/writing JSON or text must use `encoding="utf-8"` explicitly.
- Use `OK` / `WARN` / `FAIL` in status messages -- never Unicode symbols.
- NEVER use em-dashes, en-dashes, or any Unicode dashes anywhere -- code, print statements, comments, strings, or file output. Always use `--` instead. wandb intercepts stdout with its own writer that ignores PYTHONIOENCODING and uses the system locale directly.
- `PYTHONIOENCODING=utf-8` is set in `~/.bashrc` on the cluster, but SLURM jobs don't source `.bashrc`. Every sbatch script must include `export PYTHONIOENCODING=utf-8` explicitly. Don't rely on the bashrc setting -- guard in both code and sbatch.

## Shell commands

- Always single-line. No backslash continuations -- they break when pasted.
- Use `&&` or `;` to chain commands on one line.

## Training code

- Always cast `pixel_values` to the model's dtype before forward pass: `pixel_values.to(next(model.parameters()).dtype)`. Float32/bfloat16 mismatches cause silent corruption or crash.
- SLURM sbatch scripts must use `. ~/miniconda3/etc/profile.d/conda.sh` (POSIX dot-source), not `source` (requires bash).
- Always `export PYTHONPATH="$PWD:$PYTHONPATH"` in sbatch scripts.
- Always `export PYTHONUNBUFFERED=1` in sbatch scripts so stdout/stderr flush immediately. Without this, Python and wandb buffer output and job logs appear empty until the process exits.
- Request `gpu:ampere:1` for eval/training jobs -- generic `gpu:1` may get a 12GB GPU.

## Communication style

- Be objective. Push back when ideas have flaws. Don't soften critiques to preserve rapport.
- When two options are both reasonable, name the decisive tradeoff rather than forcing a preference.

## Maintaining this file

When a new convention, gotcha, or rule emerges during a session -- especially from a bug that wasted time -- add it to this file before the session ends. Examples of things that belong here:
- A new cluster environment or dependency constraint
- A data format rule that caused a silent bug
- A sbatch flag that's required but easy to forget
- A naming convention the user established
- Any "we should never do X again" moment

If you're unsure whether something belongs here, add it. A redundant line is cheaper than a repeated mistake.

## Model eval configurations -- PINNED

Each model requires a specific conda env and GPU. Always pass `CONDA_ENV=` explicitly when submitting jobs -- this skips the sbatch's unreliable transformers version heuristic and uses whatever is already installed in the env.

| Model | HF Path | CONDA_ENV | GPU | Notes |
|---|---|---|---|---|
| InternVL2.5-8B | OpenGVLab/InternVL2_5-8B | vlm_py312 | A6000 (48GB) | |
| InternVL2.5-78B-AWQ | OpenGVLab/InternVL2_5-78B-AWQ | vlm_py312_tf4451 | A100 (80GB) | needs `gpumem:80G` |
| InternVL3-9B | OpenGVLab/InternVL3-9B | vlm_py312 | A6000 (48GB) | |
| InternVideo2.5-8B | OpenGVLab/InternVideo2_5_Chat_8B | vlm_py312 | A6000 (48GB) | |
| LLaVA-Video-7B | lmms-lab/LLaVA-Video-7B-Qwen2 | vlm_py312 | A6000 (48GB) | |
| VideoLLaMA3-7B | DAMO-NLP-SG/VideoLLaMA3-7B | vlm_py312 | A6000 (48GB) | |
| Ovis2.5-9B | AIDC-AI/Ovis2.5-9B | vlm_py312 | A6000 (48GB) | THINKING_BUDGET=0 for non-thinking |
| Qwen3-VL-8B | Qwen/Qwen3-VL-8B-Instruct | vlm_py312_fromsrc | A6000 (48GB) | |
| Qwen2.5-VL-7B | Qwen/Qwen2.5-VL-7B-Instruct | vlm_py312 | A6000 (48GB) | |
| Qwen2.5-VL-72B-AWQ | Qwen/Qwen2.5-VL-72B-Instruct-AWQ | vlm_py312_tf4451 | A100 (80GB) | needs `gpumem:80G` |
| gemma-4-26B | google/gemma-4-26b-a4b-it | vlm_gemma4 | A100 (80GB) | needs `gpumem:80G`, 128G RAM; MoE 25.2B total/3.8B active |

Env contents: `vlm_py312` has transformers 4.51.3, `vlm_py312_tf4451` has 4.45.2, `vlm_py312_fromsrc` has 5.3.0, `vlm_gemma4` has 5.8.0. All envs have `flash-attn==2.7.4.post1` (pre-built wheel, cu12, cxx11abiFALSE).

## Flash Attention 2

All training and eval scripts use `attn_implementation="flash_attention_2"` in `AutoModel.from_pretrained()`. flash-attn 2.7.4.post1 is installed in all 4 conda envs via pre-built wheel. If a new env is created, install with:
`pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"`

InternVL detects flash-attn at import time and silently falls back to eager attention if the import fails -- no warning. Verify with: `python -c "from flash_attn import flash_attn_func; print('OK')"`

Job splitting: if the question file has more than 1000 questions, split it into thirds and submit each third as a separate job. Use `--part` / `--total-parts` flags or split the question file. This keeps individual job runtimes manageable and reduces the blast radius of failures.

Key gotchas:
- transformers >=4.50 removes `GenerationMixin` from `PreTrainedModel` -- breaks InternVL/InternVideo `generate()`. This is why they must use `vlm_py312` (4.49.0), not a newer env.
- InternVL2.5-78B-AWQ needs 80GB VRAM (`gpumem:80G`). Without it, the model offloads to CPU and AWQ crashes.
- The sbatch has a transformers version heuristic that will try to install a different version if it doesn't match. Passing `CONDA_ENV=` explicitly prevents this -- the env already has the right version.
- Ovis thinking vs non-thinking use the same HF path; differentiated by `THINKING_BUDGET=512` (default) vs `THINKING_BUDGET=0`.

## Model config authority

- `dispatch_all_models.sh` is the authoritative source for model configs (HF paths, transformers versions, conda envs, GPU tiers, extra pips). Always check it first before submitting jobs.
- The transformers version heuristic was removed from `all_model_multi_gpu.sbatch` -- we rely solely on `CONDA_ENV=` to avoid concurrent jobs corrupting shared conda envs via pip installs.

## Social appropriateness grading

- Grading uses `Qwen/Qwen2.5-7B-Instruct` via the `transformers` pipeline backend (`--judge transformers`), not gpt-oss via vLLM. This avoids vLLM startup overhead and GPU contention.
- The `gptoss` conda env has vLLM 0.20.0 (mainline, not the obsolete `+gptoss` wheel) if vLLM grading is ever needed.
- gpt-oss-20b requires Ampere+ GPU (compute capability >= 80). The old `gpu:1` gres got Titan Xp (cap 6.1) which crashed.

## Merging eval results

- `analysis_scripts/merge_eval_results.py` is the general-purpose merge script. Handles any question type, deduplicates by (video_name, prompt, question_type), and can output a CSV row with `--csv`. Use this for backfilling missing questions or merging results from collaborators.
- `analysis_scripts/merge_actagg_results.py` is the older Action+Aggressor-specific merge script. Still works but `merge_eval_results.py` supersedes it.
- Both handle our eval format (`is_correct`/`model_selected_index` pre-computed) and simplified format (`model_response` as raw string like `"4."` or `"D"` -- parsed automatically).
- Both automatically remove excluded videos (tackle/bodyslam/indecent gesture) and prevent duplication. Safe to run multiple times.

## Primary/secondary question type grouping

- Authoritative source: `SECONDARY_QUESTION_TYPES` in `prompt_generator/templates.py`.
- Primary (15,004 questions): primary_action, role_identification, aggressor_identification, victim_recognition, compound_action_aggressor, compound_action_victims, compound_aggressor_victim, compound_aggressor_action_victim, sequence_verification.
- Secondary (3,744 questions): compound_action_location, role_count_victim, role_count_aggressor, role_count_bystander, compound_aggressor_victim_count.
- Action+Location is SECONDARY despite being a compound type -- it was explicitly added to `SECONDARY_QUESTION_TYPES` in templates.py.
- When updating merge scripts or analysis code, always match this grouping. Getting it wrong silently corrupts Primary/Secondary accuracy columns.

## Adaptive DPO pipeline

- Training pipeline with adaptive rejection selection: SFT -> CoT-SFT -> eval-on-train (Phase 3.5) -> adaptive pair extraction -> ADPO -> eval.
- Phase 3.5 evals the CoT-SFT checkpoint on training questions to get per-question correctness.
- `extract_pairs.py --eval-results <path> --selection-strategy hard_mining` uses correctness to pick easy rejecteds for correct questions, hard rejecteds for wrong questions.
- Experiment configs: `train_model/experiments/{5,10,20}pct_adaptive_v1/`.
- Submit with: `bash train_model/experiments/Xpct_adaptive_v1/sbatch/submit_all.sh`

## Naming conventions

- ADPO = Anchored DPO (alpha > 0). DPO = vanilla DPO (alpha = 0). Keep them distinct in configs, output directories, and result files.
- Training pipeline phases: 0 (frames), 1 (data split), 2 (SFT), 3 (CoT distillation), 3.5 (eval-on-train for adaptive), 4 (CoT-SFT / pair extraction), 5 (ADPO/DPO), 6 (eval).

## Question bank and frequency-inverted distractors

- `train_model/data/generated_questions.json` is the authoritative question bank (18,748 questions, 2,670 videos). Format: `{"metadata": {...}, "questions_by_video": {"video.mp4": [questions]}}`. Tackle, bodyslam, and indecent gesture videos have been removed from this file.
- The question bank uses `hardness_profile: frequency_inverted`. The `generated_questions_freq_inv_part*.json` files are identical splits of the same data -- there is no separate "non-frequency-inverted" version.
- Frequency-inverted construction only affects 2 question types: `compound_aggressor_action_victim` and `sequence_verification`. These get 1 role_reversal + 6 frequency_saturation distractors with mathematically balanced person/action frequencies.
- All other question types use standard hardness recipes (role_reversal, wrong_action, cross_video, etc.) regardless of the profile.
- MCQ option format is `A) option text` (parenthesis), not `A. option text` (dot). Match this in any new eval or prompting code for consistency with the existing eval infrastructure.

## No-train method (RAP)

- `no_train_method/` contains a training-free retrieval-augmented 1-shot prompting baseline.
- `retriever.py`: dictionary-based lookup indexed by `(question_type, is_trick)`, parent-video-aware filtering via `dataset.json`.
- `run_eval.py`: loads base model (no adapter), retrieves 1-shot reference per test question, injects into prompt, supports checkpointing and `--part`/`--total-parts`.
- Cannot match on action or roles (data leakage). Only question_type and distractor_type are allowed matching features.

## Eval checkpoint bug (fixed)

- The eval script (`train_model/eval/run_evaluation.py`) had a bug where the checkpoint path was shared across parallel `--part` jobs. This caused later parts to accumulate results from earlier parts via the shared checkpoint file, inflating `by_question_type` totals.
- Fixed by making checkpoint paths part-aware: `.checkpoint_{stage}_part{N}of{M}.json`.
- Use `analysis_scripts/dedup_eval_results.py` to deduplicate results from runs that used the old code. It reads detailed result files, deduplicates by (video_name, prompt), and writes a corrected `results_merged.json`.
- All results in `pipeline_comparison.csv` have been deduplicated.

## Standalone vs chained experiments

- `train_model/experiments/Xpct/` = standalone baseline (SFT, CoT, DPO, ADPO each trained independently from base model, no chaining).
- `train_model/experiments/Xpct_adaptive_v1/` = chained adaptive pipeline (SFT -> CoT -> eval-on-train -> hard_mining pairs -> ADPO).
- `train_model/experiments/20pct_curriculum_v1/` = chained curriculum pipeline (inverse of hard_mining: easy rejected for wrong, hard rejected for correct).
- `train_model/experiments/5pct_adaptive_standalone/` = standalone ADPO using adaptive pairs but trained from base (no SFT warmup).
- Key finding: standalone DPO/ADPO from base scores ~50% (random). Preference optimization requires SFT warmup to work.

## Dataset counts

- Annotations: 2,683 clips (after removing tackle/bodyslam/indecent gesture). 2,234 aggressive + 449 non-aggressive ("none").
- Question bank: 2,670 videos (13 clips skipped by question generator due to quality control). 18,748 total questions (15,004 primary + 3,744 secondary).
- 17 aggressive action categories + 1 "none" control = 18 total.
- 14 question types total (9 primary + 5 secondary).

## Standard model list

These are the final models used in all charts and analysis. Analysis scripts should use this list.

**Baseline models (12):**
- gemma-4-26B-A4B-it
- InternVideo2.5-8B
- InternVL2.5-8B
- InternVL3-9B
- InternVL3.5-8B
- LLaVA-Video-7B-Qwen2
- Ovis2.5-9B
- Qwen2.5-VL-7B-Instruct
- Qwen3-VL-8B-Instruct
- VideoLLaMA3-7B
- Qwen2.5-VL-72B-Instruct-AWQ
- GPT5.1

**Variants & prompt configurations:**
- Qwen3-VL-8B-Thinking vllm
- InternVL3.5-8B detailed cot prompt (Chain of Thought)
- InternVL3.5-8B dream of thoughts

## Git rules

- Do not make any references to yourself or references to having helped with the code in commit messages
- Never force push to main without verifying with me first that I want to do that. 
