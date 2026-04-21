# ADPO Training Plan — Multimodal Video-Aggression Recognition

End-to-end plan for fine-tuning InternVL2.5-8B to identify aggressive
behaviour in short bullying-incident videos (aggressor, victim, action,
location, etc.) using a six-phase pipeline that culminates in Anchored
Direct Preference Optimization (ADPO).

The pipeline lives entirely under `train_model/`. Every file referenced
below is rooted there.

---

## 1. Objective

Given a short video clip of a real-world altercation, the model must
answer multiple-choice and free-form questions about the scene grounded
**in the frames themselves**, not in human-written annotations:

* "Who is the aggressor?"
* "What action took place?"
* "Where did this happen?"
* Compound questions combining role, action, and victim/bystander
  identity.
* Trick questions where the apparent aggressor is actually the victim.

Success means the model recovers the right structured answer from raw
visual evidence, even when the question is adversarial.

---

## 2. Base Model

| Component | Choice |
|---|---|
| Backbone | `OpenGVLab/InternVL2_5-8B` (`InternVLChatModel`) |
| Vision encoder | `InternViT-300M` ⟶ frozen for the entire pipeline |
| Vision→text projector | `mlp1` (LayerNorm + 2× Linear + GELU) ⟶ LoRA-adapted |
| Language model | `InternLM2-7B` (`InternLM2ForCausalLM`) ⟶ LoRA-adapted |
| Conversation template | `internvl2_5` (`<|im_start|>` ... `<|im_end|>`) |
| Image input | 8 frames per video, each resized to 448×448, ImageNet normalised |
| Visual tokens per frame | 256 (`num_image_token`) — total 2,048 for 8 frames |
| Precision | bf16 throughout, with gradient checkpointing |

LoRA configuration (shared across SFT, CoT-SFT, and DPO):

```yaml
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  bias: none
  target_modules:
    - wqkv     # InternLM2 attention QKV
    - wo      # InternLM2 attention output
    - w1      # InternLM2 MLP gate
    - w2      # InternLM2 MLP down
    - w3      # InternLM2 MLP up
    - mlp1.1  # vision→text projector linear 1
    - mlp1.3  # vision→text projector linear 2
```

Total trainable: ~19 M parameters (0.23 % of the full 8 B model).

`task_type` is intentionally **not** set on `LoraConfig` because
`PeftModelForCausalLM` injects an `inputs_embeds=...` kwarg that
`InternVLChatModel.forward` does not accept; the plain `PeftModel`
forward is a clean pass-through.

---

## 3. Data Architecture

### 3.1 Source artefacts

| File | Purpose | Tracked? |
|---|---|---|
| `videos/` | ~2,624 raw `.mp4` clips on the cluster | not in git |
| `annotations.json` | Per-video labels: aggressor, victim, action, environment, bystanders | yes |
| `train_model/data/generated_questions.json` | Multi-choice questions per video, produced upstream by `prompt_generator/` | not in git |

### 3.2 Splits — video-level, not question-level

`train_model/sft/format_data.py` produces `sft_train.json`,
`sft_val.json`, `sft_test.json` by:

1. Grouping all questions by `video_name`.
2. Splitting **videos** 80/10/10, stratified on each video's `action`
   label (22 distinct actions) so every split contains a comparable
   mix of aggression types.
3. Routing every question for a given video into the same split.

This eliminates the video leakage that previously inflated test
accuracy in the text-only iteration. Current split sizes:

| Split | Videos | Examples |
|---|---:|---:|
| train | 2,084 | 10,509 |
| val | 254 | 1,308 |
| test | 279 | 1,412 |

### 3.3 Frame cache (Phase 0)

`train_model/common/frame_cache.py`, invoked by
`train_model/sbatch/00_extract_frames.sbatch`:

* Reads every `video_name` referenced in the splits.
* Calls `ffprobe` to determine duration, then `ffmpeg` to grab 8
  evenly-spaced JPEG frames at `t = duration · i/(N+1)` for
  `i ∈ {1..8}`.
* Three-attempt fallback per frame: keyframe-accurate input seek →
  output-side seek with `-err_detect ignore_err` → keyframe-only
  selection. Captures stderr from each attempt.
* Writes `train_model/data/frames/{video_stem}/frame_{i:02d}.jpg`.
* Writes `_failures.json` listing every video that didn't reach status
  `ok`/`skipped`. **Always exits 0** so downstream training can
  proceed on whatever cached.
* Statuses: `ok`, `skipped`, `missing` (file absent), `corrupt`
  (ffprobe couldn't parse duration), `partial` (≥1 ffmpeg attempt
  failed).

Current cache: 2,617 videos × 8 frames = 20,936 JPEGs. After
removing 7 unusable annotations (1 blank `file_name` + 6 missing
UCF-Crime clips), Phase 0 reports zero failures.

### 3.4 Prompt format (training & inference)

The model only ever sees:

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
Frame 1: <image>
Frame 2: <image>
…
Frame 8: <image>

Question: {prompt}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

Each `<image>` is expanded to
`<img>` + `<IMG_CONTEXT>` × `num_image_token` × `num_patches` +
`</img>` so the visual-token count matches what
`extract_feature(pixel_values)` produces.

The `video_context` annotation string (the human-written
"Aggressor: …\nVictim: …\nAction: …" summary) is **deliberately
omitted** at every stage so the model cannot short-circuit the vision
path by reading the annotator's description.

### 3.5 SFT loss masking

Inside `train_model/common/video_dataset.py::VideoSFTDataset`, the
prompt is tokenised twice — once with `assistant_content=None`
(prompt-only prefix), once with the full sequence — and `labels` are
set to `-100` for every token inside the prompt prefix and for every
padding token. Loss is computed only on the assistant's answer
tokens.

---

## 4. Pipeline Phases

```
┌────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│Phase 0 │→│ Phase 1  │→│Phase 2 │→│ Phase 3  │→│Phase 4  │→│ Phase 5  │→│ Phase 6  │
│ frames │  │   SFT    │  │  CoT   │  │ CoT-SFT  │  │  DPO    │  │   ADPO   │  │   eval   │
│        │  │ baseline │  │  data  │  │  +reason │  │  pairs  │  │ training │  │  ablate  │
└────────┘  └──────────┘  └────────┘  └──────────┘  └─────────┘  └──────────┘  └──────────┘
   ffmpeg     LoRA SFT     teacher    LoRA SFT      heuristic    DPO loss +     compare
   one-time   2 epochs     reasoning  resumes       hardness     anchor term    SFT vs CoT
   ~15 min    ~5 hr        ~11 hr     Phase 1       ranking      α=0.5          vs ADPO
                           GPU                      few sec      ~6 hr          ~6 hr
```

### Phase 1 — SFT baseline (`train_model/sft/train.py`)

* Input: `(8 frames, question)`. Output: `correct_answer`.
* HF `Trainer` with the multimodal collator
  (`pixel_values`, `input_ids`, `attention_mask`, `image_flags`,
  `labels`).
* 2 epochs, per-device batch 1, gradient accumulation 8 (effective
  batch 8), LR 2e-4, cosine schedule, 200-step warmup.
* Eval every 500 steps on `sft_val.json`; best checkpoint by
  `eval_loss` is loaded at end and saved as
  `train_model/models/sft/`.
* sbatch: `train_model/sbatch/01_sft.sbatch` (24 h cap; expect ~5 h).

### Phase 2 — CoT data generation (`train_model/cot/generate_chains.py`)

* Use a strong teacher (a separate InternVL2.5-8B inference run, or
  GPT-4o, depending on what's wired up) to produce a chain-of-thought
  reasoning trace for each compound question in `sft_train.json`.
* Output: `train_model/data/cot_chains.json`, ~5,400 examples (one
  per compound train question), each carrying a `reasoning` field
  alongside the existing answer.
* sbatch: `train_model/sbatch/02_cot_data.sbatch`.

### Phase 3 — CoT-SFT (`train_model/cot/train.py`)

* Continues from the Phase 1 LoRA adapter
  (`resume_from: train_model/models/sft`).
* Same training recipe as Phase 1 but the answer text now optionally
  includes a `Reasoning: …\n\nAnswer: …` prefix for compound
  questions. Simple questions keep the bare answer to avoid
  unnecessary CoT overhead.
* Saves `train_model/models/cot/`.
* sbatch: `train_model/sbatch/03_cot.sbatch`.

### Phase 4 — Preference-pair extraction (`train_model/dpo/extract_pairs.py`)

#### Question format

Every training question already comes with a multiple-choice answer
list. The distribution from `generated_questions.json` is:

| # Options | Question count | Question types |
|---:|---:|---|
| 4 | 900 | `role_identification` only |
| 8 | 12,329 | everything else |

So a typical question has **7 distractors** (8 options minus the
correct one) available for DPO.

#### What the extractor does

For each train-split question (skipping `is_secondary` and
`is_trick`):

1. `chosen` = the correct answer; if a reasoning chain exists in
   `cot_chains.json` for that `(video_name, question_index)` pair, it
   is attached.
2. Build `distractors = all answers except the correct one` (up to 7).
3. Heuristically classify each distractor: `role_reversal`,
   `bystander_substitution`, `wrong_action`, or `cross_video`.
4. **Drop** everything classified as `cross_video`.
5. Sort the survivors by priority
   `role_reversal > bystander_substitution > wrong_action > other`.
6. Keep the top `num_rejected_per_chosen` (default **3**). Questions
   whose survivors list is empty produce a pair with `rejected=[]`
   and are skipped by `VideoDPOPairDataset` at load time.

The `num_rejected_per_chosen` cap is a tractability knob: each DPO
optimiser step already runs 4 forward passes through InternViT +
LLM (2 chosen + 2 rejected, with the reference policy recomputed by
`peft.disable_adapter()`), and adding more rejected responses per
pair multiplies that cost.

Restricted to videos in `sft_train.json` so DPO never sees pairs
built from val/test videos.

Output: `train_model/data/preference_pairs.json` (≈1,800 pairs in
the previous run).

sbatch: `train_model/sbatch/04_extract_pairs.sbatch`.

#### Known limitation — the classifier is under-performing

The current `classify_distractor()` heuristic was written expecting
distractors to carry explicit role tags like `"… (aggressor)"` and
`"… (victim)"`. The actual distractors are free-form prose, so the
heuristic falls through to `cross_video` for essentially every
compound-question distractor — including the genuinely hard ones
(role reversals and wrong-action same-cast substitutions). The
result is that the vast majority of 8-option compound pairs end up
with `rejected = []` and are silently discarded.

Concrete example from `punch_facebook_003.mp4`
(`compound_aggressor_action_victim`, 8 options):

```
CORRECT: person in a green shirt and dark pants performed punch on person in light pants

Distractor                                                            Current label      In-cast?
--------------------------------------------------------------------- ------------------ --------
person in light pants performed punch on person in a green shirt…     cross_video  DROP  YES (role reversal)
person in a green shirt and dark pants performed choke on person…     cross_video  DROP  YES (wrong action, same cast)
person wearing blue shirt performed shove on person in white top…     cross_video  DROP  no
…
```

All 7 distractors are dropped, leaving an empty rejected list; the
pair is discarded.

Even when one distractor survives, the filter can keep the easier
one and throw away the harder one. Example from
`punch_chatgpt_026.mp4` (`role_identification`, 4 options):

```
CORRECT: Aggressor
  "No one in the video fits that description"  ->  cross_video  DROP  (fine, truly weak)
  "Victim"                                      ->  cross_video  DROP  ← actually a role reversal!
  "Bystander"                                   ->  bystander_substitution  KEEP
```

The bystander contrast survives; the role-reversal contrast — the
one that would actually test whether the model can tell who attacked
whom — does not.

Two follow-ups are therefore on the table before Phase 5 trains:

1. **Rewrite the classifier** to detect role reversal and
   wrong-action substitutions semantically (e.g. parse the
   aggressor / action / victim slots out of the distractor string
   and compare against the correct tuple).
2. **Bump `num_rejected_per_chosen`** from 3 to 5 or 7 so that all
   surviving hard distractors make it into training, not just the
   top three by priority.

Both changes are safe to make after Phase 1 SFT finishes; Phase 4
itself runs in seconds.

### Phase 5 — (A)DPO (`train_model/dpo/train.py`)

The headline phase. For each preference pair `(prompt, chosen, rejected)`
both responses are encoded against the same `pixel_values` and the
loss is

```
margin   = β · ((logπ_pol(chosen) − logπ_ref(chosen)) − (logπ_pol(rejected) − logπ_ref(rejected)))
L_dpo    = −logσ(margin).mean()
L_anchor = mean( (logπ_pol(chosen) − logπ_ref(chosen))² + (logπ_pol(rejected) − logπ_ref(rejected))² )
L_total  = L_dpo + α · L_anchor
```

where each `logπ_*` is the **sum of log-probs over response tokens
only** (not the shared prompt). The reference policy is the same
model with the LoRA adapter disabled via `peft.disable_adapter()`,
so there is no second 8 B copy on the GPU.

Defaults (see `train_model/configs/dpo.yaml`):

| Hyper | Value | Notes |
|---|---|---|
| `dpo.beta` | 0.1 | Standard DPO temperature |
| `dpo.alpha` | 0.5 | ADPO anchor weight (set 0 ⇒ vanilla DPO) |
| epochs | 1 | Default; preference data is small |
| effective batch | 4 | per-device 1 × accum 4 |
| LR | 5e-6 | Small — DPO is sensitive |
| warmup | 3 % of total steps | Linear |
| gradient checkpointing | on | Memory pressure: 4 forward passes/step |

The capability for vanilla DPO is preserved — flip
`--override dpo.alpha=0.0`. Sweeps:

```
for a in 0.0 0.1 0.3 0.5 1.0; do
  sbatch train_model/sbatch/05_dpo.sbatch \
    --override dpo.alpha=$a \
    --override output_dir=train_model/models/dpo_a$a
done
```

Saves `train_model/models/dpo/` (or `dpo_a*` for sweeps).

### Phase 6 — Evaluation & ablation (`train_model/eval/run_evaluation.py`)

* Loads each saved adapter (SFT / CoT-SFT / DPO) onto the **full**
  `InternVLChatModel`, not the language-model submodule, so adapter
  keys match the saved paths.
* Drives `model.chat(tokenizer, pixel_values, question, ...)` with
  the same 8-frame prompt format used at training time.
* Greedy decode, max 64 new tokens. Exact-match or substring-contains
  scoring against `correct_answer`.
* Aggregates accuracy overall, by `question_type`, and by
  `is_trick`.
* Writes `train_model/eval/results.json` and prints an ablation
  summary.
* sbatch: `train_model/sbatch/06_evaluate.sbatch`.

---

## 5. Configuration System

YAML configs live in `train_model/configs/` and merge through a small
`defaults:` mechanism implemented in `train_model/common/config.py`.

```
train_model/configs/
  base.yaml          # model name, video, lora, tokenization, seed
  sft.yaml           # Phase 1 overrides       (defaults: [base])
  cot.yaml           # Phase 3 overrides       (defaults: [base])
  dpo.yaml           # Phase 5 overrides       (defaults: [base])
  eval.yaml          # Phase 6 overrides       (defaults: [base])
  sft_smoke.yaml     # tiny 100-step Phase 1   (defaults: [base])
```

Every CLI accepts `--override key.path=value` (parsed as YAML so types
land correctly) so quick experiments don't require editing files. The
fully-merged config is written to `{output_dir}/config.yaml` at the
start of every training run for reproducibility.

---

## 6. Reproducibility & Seeds

* Seed = 42 throughout (`transformers.set_seed`, `torch.manual_seed`,
  `random.seed` in data prep).
* All YAML files are opened with `encoding="utf-8"`; the cluster's
  default locale is latin-1 and bare `open()` calls would otherwise
  break on em-dashes in comments.
* All ASCII-only output strings — UTF-8 special characters
  (α, ✓, →, etc.) have been removed from `print()` calls because
  Python's default stdio encoding is latin-1 on the cluster.

---

## 7. Hardware & Throughput

Single A6000 (48 GB) or A100 (40 GB) per phase.

| Phase | Wall-clock (estimated) | GPU mem | Notes |
|---|---:|---:|---|
| 0 frames | ~15 min | n/a | CPU partition `short` |
| 1 SFT | ~5 h | ~25 GB | 2 epochs × 2,627 opt steps × ~7 s/step |
| 2 CoT data | ~11 h | ~30 GB | Teacher inference, dominates by token count |
| 3 CoT-SFT | ~5 h | ~25 GB | Same shape as Phase 1 |
| 4 pairs | < 1 min | n/a | Pure CPU/Python |
| 5 ADPO | ~6 h | ~30 GB | 4 forward passes/step (2 ref no-grad + 2 policy) |
| 6 eval | ~3 h | ~25 GB | Greedy decode × 1,412 test × 3 stages |

---

## 8. Smoke-Test Validation

Before kicking off the real Phase 1 run, the smoke config
(`train_model/configs/sft_smoke.yaml`) trains for **100 optimizer
steps** on the val split with `eval_steps=50`. Most recent run
(job 337859):

* Status: COMPLETED in 47 m 7 s on an A6000.
* Trainable params: **19,005,440** (0.23 %).
* Train loss: **3.57 → 0.68** (intermediate min) → ~1.0 at final step.
* **Eval loss: 1.18 → 0.89** between the two eval checkpoints. ← real
  signal that frames are doing the work.
* Adapter saved cleanly (no `base_model.model.model` key path
  mismatch).
* Peak GPU memory ≈ 19 GB on a 49 GB card.

This validates that:

1. The multimodal forward path (vision tower → mlp1 → InternLM2 →
   loss) is working end-to-end.
2. Gradients flow back through the LoRA adapters on **both** the LLM
   layers and the projector.
3. The model is actually learning *something* from the frames, not
   just from leaked annotations (which are now absent).
4. Adapter checkpoints are reload-compatible by the eval script.

---

## 9. Known Caveats

* **Frozen vision encoder.** We do not LoRA-adapt InternViT, on the
  bet that its ImageNet/CLIP-pretrained features are good enough and
  that adapting `mlp1` is sufficient to bridge the domain gap. If
  Phase 6 shows the model is bottlenecked on visual perception
  rather than reasoning, we may need to add LoRA on
  `vision_model.encoder.layers.*` later.
* **8 frames only.** Long videos may contain critical action outside
  these eight evenly-spaced samples. If the dataset includes clips
  longer than ~30 s and accuracy on those is poor, bump
  `video.frames_per_video` (and re-run Phase 0).
* **Two videos with one cropped frame each will not be flagged
  partial** — we only check that frame jpgs exist on disk, not that
  they decoded cleanly. If we suspect a few examples are corrupt-but-
  cached, add a Pillow open-and-load check inside the dataset.
* **Preference-pair imbalance.** Phase 4's hardness distribution is
  heavily skewed toward `bystander_substitution` and
  `role_reversal`; only a handful of `wrong_action` pairs exist
  (because the question generator rarely produces that distractor
  class). The DPO model will learn role/bystander discrimination
  much better than action discrimination.
* **`_failures.json` is informational only.** It is not consumed by
  any downstream script — the dataset filters at load time by
  checking `_frames_cached(...)`. Manually inspect the manifest if
  Phase 0 reports failures.

---

## 10. Quick Start

Assuming the working directory is the repo root and PYTHONPATH
includes it:

```bash
# Phase 0 (one-time, ~15 min)
sbatch train_model/sbatch/00_extract_frames.sbatch

# (regenerate splits if annotations.json or generated_questions.json changed)
python train_model/sft/format_data.py --force

# Phase 1 SFT (~5 h)
sbatch train_model/sbatch/01_sft.sbatch

# Phase 2 CoT (~11 h)
sbatch train_model/sbatch/02_cot_data.sbatch

# Phase 3 CoT-SFT (~5 h)
sbatch train_model/sbatch/03_cot.sbatch

# Phase 4 preference pairs (~seconds)
sbatch train_model/sbatch/04_extract_pairs.sbatch

# Phase 5 ADPO (~6 h, alpha=0.5 by default)
sbatch train_model/sbatch/05_dpo.sbatch

# Phase 6 evaluation (~3 h)
sbatch train_model/sbatch/06_evaluate.sbatch
```

Smoke test:

```bash
sbatch train_model/sbatch/01_sft_smoke.sbatch    # ~45 min
```

Outputs land under `train_model/models/{sft,cot,dpo}/`, eval results
in `train_model/eval/results.json`.

---

## 11. File Layout

```
train_model/
├── README.md                     # quick reference
├── ADPO_Training_Plan.md         # this file (lives at repo root, not here)
├── requirements.txt              # PyYAML, Pillow, torchvision, transformers, peft, ...
├── common/
│   ├── config.py                 # YAML loader with defaults: + dotted overrides
│   ├── frame_cache.py            # ffmpeg-based one-time extraction
│   └── video_dataset.py          # VideoSFTDataset, VideoDPOPairDataset, collators
├── configs/
│   ├── base.yaml                 # shared
│   ├── sft.yaml
│   ├── cot.yaml
│   ├── dpo.yaml
│   ├── eval.yaml
│   └── sft_smoke.yaml
├── sft/
│   ├── train.py                  # Phase 1
│   ├── format_data.py            # builds video-stratified train/val/test splits
│   └── regenerate_benchmark.py   # rebuild generated_questions.json from raw
├── cot/
│   ├── generate_chains.py        # Phase 2
│   └── train.py                  # Phase 3
├── dpo/
│   ├── extract_pairs.py          # Phase 4
│   └── train.py                  # Phase 5 (DPO + optional anchor)
├── eval/
│   └── run_evaluation.py         # Phase 6
├── sbatch/
│   ├── 00_extract_frames.sbatch
│   ├── 01_sft.sbatch
│   ├── 01_sft_smoke.sbatch
│   ├── 02_cot_data.sbatch
│   ├── 03_cot.sbatch
│   ├── 04_extract_pairs.sbatch
│   ├── 05_dpo.sbatch
│   └── 06_evaluate.sbatch
├── data/
│   ├── generated_questions.json  # input
│   ├── sft_{train,val,test}.json # output of format_data.py
│   ├── cot_chains.json           # output of Phase 2
│   ├── preference_pairs.json     # output of Phase 4
│   └── frames/                   # output of Phase 0 (8 jpgs/video)
└── models/                       # adapters land here per phase
```
