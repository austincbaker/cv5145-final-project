# How-To Guide

Operational reference for this repo. Covers three common workflows:

1. **Generate a new question set** from video annotations (with hardness
   profiles and optional parallel-eval splits).
2. **Add a new VLM** to the baseline evaluation framework.
3. **Run a baseline** against a question set, with or without video frames,
   and combine results from split runs.

---

## 1. Repository Layout at a Glance

```
bullying-project/
├── annotations.json                      SOURCE: per-video labels
├── videos/                               SOURCE: raw .mp4 clips (not in git)
├── generated_questions_*.json            GENERATED: question sets (snapshots)
│
├── prompt_generator/                     QUESTION + BASELINE EVAL package
│   ├── generator.py                      QuestionGenerator (main logic)
│   ├── templates.py                      QuestionType enum + prompt templates
│   ├── answer_bank.py                    AnswerBank: pools of people/actions/etc.
│   ├── distribution.py                   CategoryDistributor (per-video mix)
│   ├── hardness.py                       Hardness taxonomy + recipes + profiles
│   ├── mutations.py                      Fake-Entry mutators (recipe backends)
│   ├── frequency_inverted.py             Standalone inverted-layout builder
│   └── evaluation/                       Baseline evaluator (not training)
│       ├── run_evaluation.py             CLI entry for frames+text evaluation
│       ├── evaluator.py                  VideoQuestionEvaluator (inference loop)
│       ├── video_processor.py            Frame extraction + async prefetch
│       ├── parallel_runner.py            Multi-GPU subprocess coordinator
│       ├── gpu_worker.py                 Per-GPU worker subprocess
│       └── model_loader/                 Pluggable VLM registry
│           ├── base.py                   BaseVLMLoader, ModelConfig
│           ├── registry.py               model path -> loader class
│           ├── example.py                Scaffold / usage patterns
│           └── <family>.py               One file per model family
│
├── generate_questions_local.py           Top-level wrapper for question gen
├── make_frameless_questions.py           Strip secondary questions for text eval
├── questions_to_markdown.py              Human-readable dump of a question set
├── combine_eval_results.py               Merge per-part eval results
├── text_only_eval.py                     Text-only baseline (no frames)
├── extract_frames.py                     Optional: pre-extract frames to jpg
│
├── *.sbatch                              SLURM job scripts (see section 5)
├── ./$USER/results_<model>-<jobid>/      Frames+text evaluation output
│
├── other/                                Planning docs, sample question sets,
│                                         experimental outputs (untracked)
└── other_scripts/                        One-off analysis / helper scripts
```

**One-liner distinctions:**
- **Source:** `annotations.json` + `videos/`
- **Question sets:** `generated_questions_*.json` (inputs to evaluation)
- **Baseline eval code:** `prompt_generator/evaluation/` (this guide)
- **Results:** `./$USER/results_<model>-<jobid>/` for each baseline run
  (see section 5 for why output is routed under the user's subdir)

---

## 2. Generating Questions

### 2a. Prerequisites

Only `annotations.json` is required. Each annotation entry needs:
```json
{
  "file_name": "punch_facebook_003.mp4",
  "aggressor": "person in a green shirt and dark pants",
  "victim": "person in light pants",
  "action": "punch",
  "environment": "street",
  "bystanders": "a group of people"
}
```

### 2b. The CLI

```bash
python generate_questions_local.py annotations.json
# writes generated_questions_<timestamp>.json at repo root
```

Common flags:

| Flag | Default | Purpose |
|---|---|---|
| `-o, --output PATH` | `generated_questions_<ts>.json` | Explicit output path |
| `-d, --num-distractors N` | 7 | Wrong answers per question (total options = N+1) |
| `--trick-probability P` | 0.10 | Fraction of questions whose correct answer is a "none of the above" style option |
| `--sample N_OR_FRAC` | (off) | Prototype mode: `0.1` = 10 % of videos, `50` = first 50 videos |
| `--seed N` | 42 | RNG seed (reproducibility) |
| `--hardness-profile {easy,balanced,hard,custom,frequency_inverted}` | `balanced` | Difficulty mix (see 2f) |
| `--recipe PATH` | (off) | JSON file of per-qtype recipe overrides; required with `--hardness-profile custom` |
| `--split N` | 1 | Partition output across N files (videos balanced); enables parallel eval (see section 4e) |

### 2c. Output schema

```json
{
  "metadata": {
    "num_videos": 2617,
    "num_questions": 13229,
    "num_distractors": 7,
    "trick_probability": 0.10,
    "hardness_profile": "balanced",
    "split": {"part": 1, "total": 3}
  },
  "questions_by_video": {
    "punch_facebook_003.mp4": [
      {
        "video_name": "punch_facebook_003.mp4",
        "question_type": "compound_aggressor_action_victim",
        "is_trick": false,
        "is_secondary": false,
        "prompt": "Which of the following best describes what happened in the video?",
        "answers": [
          "person wearing blue shirt performed shove on ...",
          "person in light pants performed punch on ...",
          "..."
        ],
        "correct_answer": "person in a green shirt and dark pants performed punch on person in light pants",
        "correct_index": 5,
        "option_hardness": [
          "cross_video", "role_reversal", "wrong_action",
          "bystander_substitution", "cross_video",
          "correct", "wrong_victim", "wrong_aggressor"
        ]
      }
    ]
  }
}
```

New fields since the first draft:
- `option_hardness`: per-option category, length matches `answers`,
  the slot at `correct_index` is always `"correct"`.
  `train_model/dpo/extract_pairs.py` consumes this field directly when
  present and skips re-classification.
- `metadata.hardness_profile`: which profile was used (see 2f).
- `metadata.split`: present only when `--split N` was passed; records
  `{"part": i, "total": N}` so downstream tools can identify shards.

Each video gets roughly **8 questions** (2 simple, 3 compound, 1 complex,
1 counting, 1 identification). Two caveats:
- `compound_aggressor_location` is currently disabled (text-only models
  solved it via property-frequency shortcuts — see commit history).
- `compound_aggressor_victim` only appears on every other video to roughly
  halve its volume.

### 2d. Typical workflow

```bash
# 1. Prototype with a small slice to eyeball output
python generate_questions_local.py annotations.json \
    --sample 20 --seed 1 -o sample_questions.json

# 2. Dump as markdown to visually inspect distractor quality
python questions_to_markdown.py sample_questions.json
# writes sample_questions.md with per-option hardness labels and
# [INVERTED] / [TRICK] / [SECONDARY] tags in each question header

# 3. Generate the full production set (balanced profile, default)
python generate_questions_local.py annotations.json \
    -o generated_questions.json

# 4. (optional) Generate a harder variant for a stress test
python generate_questions_local.py annotations.json \
    --hardness-profile hard -o generated_questions_hard.json
```

### 2e. Adding / changing question types

Question templates live in `prompt_generator/templates.py`. To add a new
question type:

1. Add a value to the `QuestionType` enum and (if needed) a new
   `Category`.
2. Add its prompt template to the `QUESTION_TEMPLATES` dict.
3. Teach `prompt_generator/generator.py:QuestionGenerator` to build
   correct answers and distractors for the new type (look for existing
   `_build_*_answer` methods as templates).
4. Update `prompt_generator/distribution.py:CategoryDistributor` if
   you want the new type to be included in the per-video quota.
5. If the new type should get category-driven distractor construction,
   add a `HardnessRecipe` for it to
   `prompt_generator/hardness.py:DEFAULT_RECIPES`.
6. Regenerate: `python generate_questions_local.py annotations.json`.

### 2f. Hardness profiles

A hardness profile is a per-qtype `HardnessRecipe` that tells the
generator what mix of distractor categories to produce. Profiles live
in `prompt_generator/hardness.py`.

Categories available (priority order, hardest first):
`role_reversal`, `wrong_action`, `wrong_victim`, `wrong_aggressor`,
`bystander_substitution`, `wrong_location`, `wrong_category`,
`none_claim`, `other_in_cast`, `cross_video`, `frequency_saturation`.

| Profile | Behaviour |
|---|---|
| `easy` | Every distractor is `cross_video` (person + scene from another annotation). Minimum discrimination. |
| `balanced` | Default mix: in-cast mutations (role_reversal, wrong_action, ...) plus 1-2 `cross_video` slots for variety. |
| `hard` | `cross_video` slots rolled into in-cast categories (role_reversal, wrong_action, or wrong_category depending on template). Every distractor is a mutation of the actual video's cast. |
| `custom` | Caller supplies a JSON file of per-qtype overrides via `--recipe PATH`. |
| `frequency_inverted` | Composes **on top of hard** (non-target qtypes use hard). For `compound_aggressor_action_victim`, `interaction_summary`, and `sequence_verification`, generates a rigid 8-option layout that flattens the per-property frequency distribution so a text-only majority-vote heuristic cannot identify the answer. |

Custom recipe file format:
```json
{
  "compound_aggressor_action_victim": {"role_reversal": 2, "cross_video": 5},
  "primary_action": {"wrong_category": 7}
}
```
Any qtype omitted falls back to its `DEFAULT_RECIPES` entry. Recipe
counts must sum to `num_distractors` per qtype.

Full walkthrough of the taxonomy design is in `other/claude_plan.md`.

### 2g. Splitting a question set across parallel eval jobs

Use `--split N` to partition the output into balanced shards named
`<stem>_partXofN.json`. Each shard carries full metadata (plus a
`metadata.split` marker) and is self-contained for `text_only_eval.py`
or the frame-based eval pipeline.

```bash
python generate_questions_local.py annotations.json \
    --hardness-profile frequency_inverted --split 3 \
    -o generated_questions_freq_inv.json
# Writes:
#   generated_questions_freq_inv_part1of3.json
#   generated_questions_freq_inv_part2of3.json
#   generated_questions_freq_inv_part3of3.json
```

Each shard can then be submitted to its own SLURM job in parallel. After
all jobs finish, `combine_eval_results.py` merges the per-shard result
JSONs back into one summary (see section 4e).

---

## 3. Adding a New VLM to the Baseline

The baseline evaluator uses a registry-of-loaders pattern. To add a new
model, you write a loader class and plug it into the registry. No changes
to the evaluator itself are needed.

### 3a. What a loader must implement

All loaders inherit `BaseVLMLoader` from
`prompt_generator/evaluation/model_loader/base.py`:

```python
class BaseVLMLoader(ABC):
    MODEL_FAMILY: str = "base"          # identifier string
    EXTRA_PACKAGES: list[str] = []      # pip packages auto-installed on load
    UPGRADE_PACKAGES: list[str] = []    # packages to pip --upgrade on load

    @abstractmethod
    def load(self) -> None: ...
        # populate self.model and self.processor

    @abstractmethod
    def generate_response(self, images: list[PIL.Image], prompt: str,
                          **kwargs) -> str: ...

    # Inherited:
    def unload(self): ...           # free GPU + clear CUDA cache
    def get_memory_usage(self): ... # dict of allocated/reserved MB
    def warmup(self): ...           # optional: dummy forward pass
    def ensure_packages(self): ...  # auto-installs EXTRA_PACKAGES
```

Settings come from a shared `ModelConfig` dataclass. Use
`self.config.model_path`, `self._get_dtype()`, `self.config.max_new_tokens`
etc. — see `base.py` for the full field list.

### 3b. Step-by-step: adding a loader for `Foo/FooVision-7B`

1. **Create the loader module**:

   ```python
   # prompt_generator/evaluation/model_loader/foo.py
   from .base import BaseVLMLoader

   class FooVisionLoader(BaseVLMLoader):
       MODEL_FAMILY = "foo"
       EXTRA_PACKAGES = ["foo-vision-utils"]   # optional

       def load(self):
           torch = self._get_torch()
           self._setup_cuda_optimizations()
           from transformers import AutoProcessor, AutoModelForCausalLM

           self.processor = AutoProcessor.from_pretrained(
               self.config.model_path, trust_remote_code=True,
           )
           self.model = AutoModelForCausalLM.from_pretrained(
               self.config.model_path,
               torch_dtype=self._get_dtype(),
               device_map=self.config.device_map or self.config.device,
               trust_remote_code=True,
           ).eval()

       def generate_response(self, images, prompt, **kwargs):
           torch = self._get_torch()
           inputs = self.processor(
               images=images, text=prompt, return_tensors="pt",
           ).to(self.model.device)
           with torch.no_grad():
               out = self.model.generate(
                   **inputs,
                   max_new_tokens=self.config.max_new_tokens,
                   do_sample=self.config.do_sample,
               )
           return self.processor.decode(
               out[0][inputs["input_ids"].shape[1]:],
               skip_special_tokens=True,
           ).strip()
   ```

2. **Register the pattern** in
   `prompt_generator/evaluation/model_loader/registry.py:_get_loader_registry()`:

   ```python
   return [
       ...
       (["foo-vision", "foo/"], ".foo", "FooVisionLoader"),
       ...
   ]
   ```

   The first matching pattern wins — put more-specific patterns above
   generic ones.

3. **Add a CLI shortcut** (optional) in the same file:

   ```python
   MODEL_SHORTCUTS: dict[str, str] = {
       ...
       "foo-7b": "Foo/FooVision-7B",
   }
   ```

4. **Smoke-test** before submitting a full run:

   ```python
   from prompt_generator.evaluation.model_loader import create_loader
   from PIL import Image

   loader = create_loader("Foo/FooVision-7B")
   loader.load()
   frames = [Image.new("RGB", (224, 224), "red") for _ in range(8)]
   print(loader.generate_response(frames, "Describe these frames."))
   loader.unload()
   ```

   Or use the built-in harness:
   ```bash
   python prompt_generator/evaluation/model_loader/example.py basic
   ```

### 3c. Loaders already wired up

| `MODEL_FAMILY` | File | Covers |
|---|---|---|
| `internvl` | `internvl.py` | InternVL2.5 / InternVL3 (all sizes) |
| `qwen_vl` | `qwen_vl.py` | Qwen2.5-VL (native video via `qwen_vl_utils`) |
| `ovis` | `ovis.py` | Ovis2.5 (thinking-mode + budget) |
| `llava-video` | `llava_video.py` | LLaVA-Video-7B |
| `videollama` | `videollama.py` | VideoLLaMA3 |
| `kimi` | `kimi.py` | KimiVL-A3B (Moonshot) |
| `nvila` | `nvila.py` | NVIDIA VILA |
| `minicpm` | `minicpm.py` | MiniCPM-V |
| `llama_vision` | `llama_vision.py` | Meta Llama 3.2 Vision |
| `phi_vision` | `phi_vision.py` | Microsoft Phi Vision |
| `generic video` | `video_generic.py` | Fallback: VideoChat, Oryx, Valley, Video-R1, Lumian, Hunyuan, InternVideo2.5 |

When in doubt, open `example.py` — it shows basic usage, multi-model
comparison, multi-GPU with `device_map="auto"`, and runtime registration
of custom models without editing the source.

---

## 4. Running a Baseline

Two modes, two entry points:

- **Frames+text**: the model sees `N` extracted video frames plus the
  question. Entry point: `prompt_generator/evaluation/run_evaluation.py`
  (single GPU) or `all_model_multi_gpu.sbatch` (SLURM multi-GPU, typical).
- **Text-only**: the model sees only the question prompt and option list.
  No frames. Useful as a lower bound that measures how much the model
  can guess from text alone. Entry point: `text_only_eval.py`
  (standalone) or `text_only_eval.sbatch` (SLURM).

Both now use **letter-based MCQ output** (`A)`, `B)`, `C)`, ...) and the
same `parse_letter` regex for scoring, so text-only and frame-based
numbers are directly comparable when run on the same question set.

### 4a. Frames+text baseline (single GPU)

```bash
python -m prompt_generator.evaluation.run_evaluation \
    annotations.json \
    videos/ \
    -m qwen-7b \
    -n 10 \
    -f 8 \
    -o results_qwen-7b/ \
    --num-gpus 1
```

Common flags:
- `annotations_json` (positional) — the annotation file.
- `video_dir` (positional) — where the `.mp4`s live.
- `-m / --model MODEL` — shortcut (`qwen-7b`) or full HF path.
- `-n / --num-questions-per-video N` — cap questions asked per video.
- `-f / --num-frames N` — frames sampled per video (default 8).
- `-o / --output DIR` — result directory.
- `--num-gpus N` — if > 1, delegates to `parallel_runner.py` which
  spawns `gpu_worker.py` subprocesses, one per GPU, each handling a
  chunk of videos. Per-worker checkpoints land in `DIR/checkpoints/`
  and are merged into `DIR/evaluation_results.json` when all finish.

### 4b. Frames+text baseline (SLURM, multi-GPU)

`all_model_multi_gpu.sbatch` is the production launcher. It reads
configuration from env vars passed via `sbatch --export`:

```bash
# Basic run — model heuristic picks its own transformers version
sbatch --export=ALL,\
MODEL=OpenGVLab/InternVL3_5-8B,\
QUESTIONS_FILE=generated_questions.json,\
OUTPUT_DIR=./results_mymodel \
    all_model_multi_gpu.sbatch

# With explicit dependency overrides (recommended for new models)
sbatch --export=ALL,\
MODEL=Qwen/Qwen2.5-VL-72B-Instruct,\
QUESTIONS_FILE=generated_questions.json,\
OUTPUT_DIR=./results_qwen72b,\
TORCH=2.1.0,\
TRANSFORMERS=4.49.0 \
    all_model_multi_gpu.sbatch
```

Supported env vars:

| Var | Purpose |
|---|---|
| `MODEL` | Required. Model shortcut or full HF path. |
| `QUESTIONS_FILE` | Pre-generated question JSON (omit to generate on the fly). |
| `OUTPUT_DIR` | Subdir name for results. Always nested under `./$USER/` (see 4f). |
| `NUM_FRAMES` | Frames per video (default 8). |
| `NUM_GPUS` | How many GPUs to split across (default 1). |
| `TORCH` | Exact torch version to install; prefix-matched (`2.1.0` accepts `2.1.0+cu121`). Unset = leave torch alone. |
| `TORCH_INDEX` | Optional `--index-url` for torch (e.g. `https://download.pytorch.org/whl/cu121`). |
| `TRANSFORMERS` | Exact transformers version. Unset falls back to the model-name heuristic. |

`sbatch --export` accepts only **one** `--export=` argument; pass `ALL`
plus all variables in a single comma-separated list. Any install failure
aborts the job with `exit 1` so the eval doesn't silently run against
the wrong dependency versions.

### 4c. Text-only baseline

```bash
python text_only_eval.py \
    --questions-file generated_questions.json \
    --model qwen-7b \
    --output results_text_only/ \
    --limit 50         # optional cap for quick iteration
```

Or via SLURM:
```bash
sbatch --export=ALL,\
MODEL=OpenGVLab/InternVL3_5-8B,\
QUESTIONS_FILE=generated_questions.json \
    text_only_eval.sbatch
```

The prompt format matches the frame eval — options are labelled with
letters and the model is asked to respond with the option letter:
```
Question: <prompt>
Options:
A) <answer 1>
B) <answer 2>
...

Answer with the option letter (A, B, C, ...) followed by the option text.
```

Responses are parsed with `LETTER_RE = re.compile(r"\b([A-H])\b\s*[\)\.\:]?", re.IGNORECASE)`.
If no letter is found, scoring falls back to substring match against
the correct answer text. Each run's summary includes
`letter_parsed_rate` so you can see whether the model is actually
emitting letters (target ≥ 95 %).

**Normal eval parity note:** `text_only_eval.py` reads the question set
as-is, which includes secondary questions (counting, `compound_action_*`).
The frame-based path reads `sft_test.json`, which has already had those
filtered out. Run `make_frameless_questions.py` first to mirror the
frame eval's question set:

```bash
python make_frameless_questions.py generated_questions.json
# writes generated_questions_text_only.json (secondary questions dropped)
python text_only_eval.py --questions-file generated_questions_text_only.json ...
```

`make_frameless_questions.py --drop-trick` also removes trick questions
if you want a stricter primary-only set.

### 4d. Interpreting results

Both modes emit a JSON file containing per-question records:

```json
{
  "video_name": "...",
  "question_type": "...",
  "prompt": "...",
  "answers": ["..."],
  "correct_answer": "...",
  "correct_index": 5,
  "model_response": "...",
  "model_selected_index": 5,
  "is_correct": true
}
```

Plus an aggregate summary at the top level (`accuracy`,
`accuracy_by_type`, `accuracy_by_trick`, `letter_parsed_rate` for
text-only; `primary_accuracy` / `secondary_accuracy` for frame-based
parallel_runner).

Aggregate overall accuracy quickly with:
```bash
jq '.accuracy // .primary_accuracy' results_<model>-<jobid>/evaluation_*.json
```

Or slice by `question_type` with a short Python script.

### 4e. Combining results from `--split` runs

When you split generation with `--split N`, submit N jobs in parallel
and merge their result JSONs at the end:

```bash
# 1. Split the generation
python generate_questions_local.py annotations.json \
    --hardness-profile frequency_inverted --split 3 \
    -o generated_questions_freq_inv.json

# 2. Submit 3 eval jobs in parallel, one per shard
MODEL=OpenGVLab/InternVL3_5-8B
for i in 1 2 3; do
    sbatch --export=ALL,\
MODEL=$MODEL,\
QUESTIONS_FILE=generated_questions_freq_inv_part${i}of3.json,\
OUTPUT_DIR=./results_freq_inv_part${i}_${MODEL//\//_} \
        all_model_multi_gpu.sbatch
done

# 3. After all finish, merge the evaluation_*.json outputs
python combine_eval_results.py \
    ./$USER/results_freq_inv_part1_*/evaluation_*.json \
    ./$USER/results_freq_inv_part2_*/evaluation_*.json \
    ./$USER/results_freq_inv_part3_*/evaluation_*.json \
    -o combined_freq_inv.json
```

`combine_eval_results.py` auto-detects both summary formats (text-only
and parallel_runner) and re-aggregates per-type / per-trick counts
from the raw `results` lists — merged accuracies are exact, not averaged
across shards. Primary vs. secondary partitioning uses the same
question-type classifier `parallel_runner` uses (by
`SECONDARY_QUESTION_TYPES` membership in `templates.py`), so merged
numbers are directly comparable to an unsplit run.

### 4f. When to use which

- **Text-only first** when bringing up a new question set: confirms the
  questions aren't accidentally giving the answer away in the prompt.
  With the `frequency_inverted` profile targeting property-frequency
  shortcuts, text-only accuracy should trend toward random chance on
  the inverted qtypes; a much higher text-only number than chance flags
  a shortcut you missed.
- **Frames+text** for the real measurement.
- **Both in parallel** when writing up results — the text-only score is
  the floor, the frames+text score is the ceiling of what the question
  format permits; the delta is what the visual input contributes.
- **Run `--hardness-profile frequency_inverted` and `hard` side by
  side** to see how much of a frame-aware model's accuracy comes from
  avoiding property-frequency shortcuts vs. from actual grounding.

---

## 5. SLURM Cheatsheet

| Script | What it runs | Output dir |
|---|---|---|
| `run_eval.sbatch` | Single-GPU frames+text baseline | `./$USER/results_<model>-<jobid>/` |
| `all_model_multi_gpu.sbatch` | Multi-GPU frames+text (production) | `./$USER/results_<model>-<jobid>/` (or `OUTPUT_DIR` override, always under `./$USER/`) |
| `text_only_eval.sbatch` | Text-only baseline across multiple models | `./results_<model>-<jobid>_text_only/` |
| `check_models.sbatch` | Verifies model weights download/load | `check_models_<jobid>.out` |

### First-run prereq for `all_model_multi_gpu.sbatch`

Run once from the repo root before your first `sbatch` on any node:

```bash
mkdir -p "./$USER"
```

The sbatch routes its `.out` / `.err` files and the results dir under
`./$USER/` so concurrent users don't collide at the repo root. SLURM
doesn't auto-create that dir — if it doesn't exist when the job starts,
its stdout/stderr are silently lost.

`OUTPUT_DIR` is always nested under `./$USER/` regardless of what you
pass. A leading `./` or `$USER/` on the override is stripped to avoid
double-prefixing, so `OUTPUT_DIR=foo`, `OUTPUT_DIR=./foo`, and
`OUTPUT_DIR=./$USER/foo` all resolve to `./$USER/foo`.

### Cluster defaults

Edit at the top of each file: partition `gpu`, QOS `group3`, one Ampere
GPU. See `other/QOS_limits.txt` for the cap on concurrent jobs.

### Monitoring helpers

```bash
./other_scripts/monitor_job.sh <jobid>   # tail logs, watch state transitions
./other_scripts/node_check.sh            # quick GPU availability summary
squeue -u $USER
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

---

## 6. Where things live (quick answers)

- **"I need to regenerate the questions."** → section 2. Output is
  `generated_questions_*.json`.
- **"How do I make a harder question set?"** → section 2f.
  `--hardness-profile hard` or `--hardness-profile frequency_inverted`.
- **"I want to run the eval in parallel for speed."** → section 2g for
  split generation, section 4e for combining.
- **"How do I strip secondary questions from an existing JSON?"** →
  `python make_frameless_questions.py generated_questions.json`.
- **"I want to add a new model."** → section 3. Write
  `prompt_generator/evaluation/model_loader/<name>.py`, register in
  `registry.py`.
- **"The model needs a specific torch/transformers version."** →
  Pass `TORCH=...` and/or `TRANSFORMERS=...` via `sbatch --export`
  (section 4b).
- **"I want to run a baseline."** → section 4. Pick frames+text or
  text-only.
- **"Which model shortcuts exist?"** →
  `prompt_generator/evaluation/model_loader/registry.py:MODEL_SHORTCUTS`.
- **"Why is my eval output under `./$USER/` now?"** → section 5.
  Concurrent-user collision avoidance. `OUTPUT_DIR` still controls the
  inner directory name.
- **"How do I inspect a question set by eye?"** →
  `python questions_to_markdown.py generated_questions.json`. The resulting
  `.md` tags every question with `[INVERTED]` / `[TRICK]` / `[SECONDARY]`
  and shows per-option hardness labels inline.
