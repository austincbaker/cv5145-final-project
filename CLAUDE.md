# Project: Video Aggression Detection Benchmark

NeurIPS 2026 submission — benchmark + progressive fine-tuning pipeline for VLM-based video aggression detection.

## Server

- SLURM cluster: `au182598@crcv.eecs.ucf.edu`
- SSH: `sshpass -p "$UCF_SERVER_TOKEN" ssh -o StrictHostKeyChecking=no au182598@crcv.eecs.ucf.edu`
- Project path on server: `~/aggressive_behavior_project`
- Conda envs: `vlm_py312`, `vlm_py312_fromsrc`, `vlm_py312_tf4451`, `vlm_train_py312`

## Encoding — CRITICAL

The cluster runs `LC_ALL=C` (Latin-1). Any non-ASCII character in Python stdout/stderr crashes the job.

- All `print()` output must be ASCII-safe. Use `.encode('ascii', 'replace').decode()` on any string that might contain Unicode (especially model-generated text).
- All `open()` calls for reading/writing JSON or text must use `encoding="utf-8"` explicitly.
- Use `OK` / `WARN` / `FAIL` in status messages — never Unicode symbols.
- `PYTHONIOENCODING=utf-8` is set in `~/.bashrc` on the cluster, but SLURM jobs don't source `.bashrc`. Every sbatch script must include `export PYTHONIOENCODING=utf-8` explicitly. Don't rely on the bashrc setting — guard in both code and sbatch.

## Shell commands

- Always single-line. No backslash continuations — they break when pasted.
- Use `&&` or `;` to chain commands on one line.

## Training code

- Always cast `pixel_values` to the model's dtype before forward pass: `pixel_values.to(next(model.parameters()).dtype)`. Float32/bfloat16 mismatches cause silent corruption or crash.
- SLURM sbatch scripts must use `. ~/miniconda3/etc/profile.d/conda.sh` (POSIX dot-source), not `source` (requires bash).
- Always `export PYTHONPATH="$PWD:$PYTHONPATH"` in sbatch scripts.
- Request `gpu:ampere:1` for eval/training jobs — generic `gpu:1` may get a 12GB GPU.

## Communication style

- Be objective. Push back when ideas have flaws. Don't soften critiques to preserve rapport.
- When two options are both reasonable, name the decisive tradeoff rather than forcing a preference.

## Maintaining this file

When a new convention, gotcha, or rule emerges during a session — especially from a bug that wasted time — add it to this file before the session ends. Examples of things that belong here:
- A new cluster environment or dependency constraint
- A data format rule that caused a silent bug
- A sbatch flag that's required but easy to forget
- A naming convention the user established
- Any "we should never do X again" moment

If you're unsure whether something belongs here, add it. A redundant line is cheaper than a repeated mistake.

## Naming conventions

- ADPO = Anchored DPO (alpha > 0). DPO = vanilla DPO (alpha = 0). Keep them distinct in configs, output directories, and result files.
- Training pipeline phases: 0 (frames), 1 (data split), 2 (SFT), 3 (CoT distillation), 4 (CoT-SFT), 5 (ADPO/DPO), 6 (eval).

## Git rules

- Do not make any references to yourself or references to having helped with the code in commit messages
- Never force push to main without verifying with me first that I want to do that. 
