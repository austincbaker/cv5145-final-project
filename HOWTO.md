# How-To Guide

Operational reference for this repo. Covers three common workflows:

1. **Generate a new question set** from video annotations.
2. **Add a new VLM** to the baseline evaluation framework.
3. **Run a baseline** against a question set, with or without video frames.

For the LoRA fine-tuning pipeline (SFT → CoT-SFT → DPO), see
[`ADPO_Training_Plan.md`](ADPO_Training_Plan.md). That pipeline is a
separate body of code under `train_model/` and is not what this guide
covers.

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
│   ├── distribution.py                   CategoryDistributor (8 qs/video mix)
│   └── evaluation/                       Baseline evaluator (not training)
│       ├── run_evaluation.py             CLI entry for frames+text evaluation
│       ├── evaluator.py                  VideoQuestionEvaluator (inference loop)
│       ├── video_processor.py            Frame extraction + async prefetch
│       ├── parallel_runner.py            Multi-GPU subprocess coordinator
│       ├── gpu_worker.py                 Per-GPU worker subprocess
│       └── model_loader/                 Pluggable VLM registry
│           ├── base.py                   BaseVLMLoader, ModelConfig
│           ├── registry.py               model path → loader class
│           ├── example.py                Scaffold / usage patterns
│           └── <family>.py               One file per model family
│
├── generate_questions_local.py           Top-level wrapper for question gen
├── text_only_eval.py                     Text-only baseline (no frames)
├── extract_frames.py                     Optional: pre-extract frames to jpg
│
├── train_model/                          LoRA fine-tuning pipeline (separate)
│   └── ...                               See ADPO_Training_Plan.md
│
├── *.sbatch                              SLURM job scripts (see §5)
├── results_<model>-<jobid>/              Frames+text evaluation output
├── 18129_results_*_text_only/            Legacy text-only runs (historical)
├── PAPER_RESULTS/                        Curated final results for write-up
└── docs/                                 Additional documentation
```

**One-liner distinctions:**
- **Source:** `annotations.json` + `videos/`
- **Question sets:** `generated_questions_*.json` (inputs to evaluation)
- **Baseline eval code:** `prompt_generator/evaluation/` (this guide)
- **Training code:** `train_model/` (see `ADPO_Training_Plan.md`)
- **Results:** `results_<model>-<jobid>/` for each baseline run

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

### 2c. Output schema

```json
{
  "metadata": {
    "generated_at": "2026-04-21T13:05:12",
    "num_videos": 2617,
    "num_questions": 13229,
    ...
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
          ...
        ],
        "correct_answer": "person in a green shirt and dark pants performed punch on person in light pants",
        "correct_index": 5
      },
      ...
    ]
  }
}
```

Each video gets exactly **8 questions by default** (2 simple, 3 compound,
1 complex, 1 counting, 1 identification). The distribution is enforced
by `prompt_generator/distribution.py:CategoryDistributor`.

### 2d. Typical workflow

```bash
# 1. Prototype with a small slice to eyeball output
python generate_questions_local.py annotations.json \
    --sample 20 --seed 1 -o sample_questions.json

# 2. Sanity-check a few
python -c "
import json
d = json.load(open('sample_questions.json'))
for v, qs in list(d['questions_by_video'].items())[:2]:
    print(v)
    for q in qs[:2]:
        print(' ', q['question_type'], ':', q['prompt'][:80])
"

# 3. Generate the full set with the default seed
python generate_questions_local.py annotations.json \
    -o generated_questions.json
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
5. Regenerate: `python generate_questions_local.py annotations.json`.

Full walkthrough for the question-generation logic is in
[`Question_Generation_Process.md`](Question_Generation_Process.md).

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
  question. Entry point: `prompt_generator/evaluation/run_evaluation.py`.
- **Text-only**: the model sees only the question prompt and option list.
  No frames. Useful as a lower bound that measures how much the model
  can guess from text alone. Entry point: `text_only_eval.py`.

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

`all_model_multi_gpu.sbatch` is the production launcher. Edit the
`MODEL` and `NUM_GPUS` variables at the top, then:

```bash
sbatch all_model_multi_gpu.sbatch
```

Output lands in `results_<model>-<jobid>/`.

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
sbatch text_only_eval.sbatch    # edit MODEL/etc. inside first
```

The prompt it sends is:
```
Answer the following multiple-choice question.
Select ONLY the number (1, 2, 3, etc.) of the correct answer.

Question: <prompt>

Options:
1. <answer 1>
2. <answer 2>
...

Answer with ONLY the option number (e.g., '1' or '2').
```

Responses are parsed with regex against the expected option number.
Output dir: `results_text_only/` with `final_results.json` and
`checkpoints/`.

### 4d. Interpreting results

Both modes emit a JSON file containing per-question records:

```json
{
  "video_name": "...",
  "question_type": "...",
  "prompt": "...",
  "answers": [...],
  "correct_answer": "...",
  "correct_index": 5,
  "model_response": "...",
  "model_selected_index": 5,
  "is_correct": true
}
```

Aggregate overall accuracy with:
```bash
jq -s '{
    total: length,
    correct: [.[] | select(.is_correct==true)] | length
}' results_<model>-<jobid>/evaluation_results.json
```

Or group by `question_type` with a short Python script.

### 4e. When to use which

- **Text-only first** when bringing up a new question set: confirms the
  questions aren't accidentally giving the answer away in the prompt. If
  a text-only model scores > 40 % it probably means a distractor-quality
  problem, not multimodal understanding.
- **Frames+text** for the real measurement.
- **Both in parallel** when writing up results — the text-only score is
  the floor, the frames+text score is the ceiling of what the question
  format permits; the delta is what the visual input contributes.

---

## 5. SLURM Cheatsheet

| Script | What it runs | Output dir |
|---|---|---|
| `run_eval.sbatch` | Single-GPU frames+text baseline | `results_<model>-<jobid>/` |
| `all_model_multi_gpu.sbatch` | Multi-GPU frames+text (production) | `results_<model>-<jobid>/` |
| `text_only_eval.sbatch` | Text-only baseline across multiple models | `results_<model>-<jobid>_text_only/` |
| `check_models.sbatch` | Verifies model weights download/load | `check_models_<jobid>.out` |

Cluster defaults (edit at the top of each file): partition `gpu`, QOS
`group3`, one Ampere GPU. See `QOS_limits.txt` for the cap on concurrent
jobs.

Monitoring helpers:
```bash
./monitor_job.sh <jobid>    # tail logs, watch state transitions
./node_check.sh             # quick GPU availability summary
squeue -u $USER
scontrol show job <jobid>
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

---

## 6. Where things live (quick answers)

- **"I need to regenerate the questions."** → §2. Output is
  `generated_questions_*.json`.
- **"I want to add a new model."** → §3. Write
  `prompt_generator/evaluation/model_loader/<name>.py`, register in
  `registry.py`.
- **"I want to run a baseline."** → §4. Pick frames+text or text-only.
- **"I want to fine-tune, not just evaluate."** → See
  [`ADPO_Training_Plan.md`](ADPO_Training_Plan.md); the code is in
  `train_model/` and does not touch `prompt_generator/evaluation/`.
- **"Which model shortcuts exist?"** →
  `prompt_generator/evaluation/model_loader/registry.py:MODEL_SHORTCUTS`.
- **"Where are old results?"** → `results_*` dirs in the repo root;
  `18129_results_*_text_only/` are legacy runs from the original
  evaluation job and are kept for reference only.
