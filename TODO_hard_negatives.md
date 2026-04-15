# TODO — Hard Negative Improvements for ADPO Preference Data

**Context:** Plan 2 (SFT + CoT + ADPO) in `implementation_plans.md` assumes ~40-60K hard preference pairs extracted from the existing distractor pipeline. Analysis of `generated_questions_20260327_201047.json` and `annotations.json` shows the current dataset produces only ~9,300 hard negatives across the four role-sensitive question categories, with just **1,949 role-reversal pairs** — the most important hard-negative subtype for the dominant failure mode.

This file tracks the work needed to close that gap before committing to Plan 2 at full scale.

---

## Analysis scripts (already exist)

- `analyze_distractor_quality.py` — classifies every distractor in the generated questions file as hard (in-cast) vs. easy (cross-video filler), split by hard-negative subtype. Run it after any generator change to measure impact.
- `analyze_annotation_potential.py` — projects the *maximum achievable* hard-negative count from the raw annotations, given the proposed generator fixes. Use it to sanity-check that a generator change could actually improve things before implementing it.

---

## Focus categories

These four question types produce the preference pairs ADPO needs. All numbers are primary questions only.

| Category | Current hard % | Current hard count | Dominant failure mode |
|---|---|---|---|
| `compound_aggressor_victim` | 28.8% | 4,127 | role reversal |
| `compound_aggressor_action_victim` | 29.7% | 2,528 | role reversal + wrong action |
| `aggressor_identification` | 12.4% | 1,325 | cross-video filler |
| `victim_recognition` | 12.2% | 1,277 | cross-video filler |

---

## Tasks — ordered by leverage

### P0 — Inject role reversals into the distractor generator

- [ ] In `prompt_generator/answer_bank.py`, add a per-video role-reversal injection for every question with a named aggressor and victim:
  - `compound_aggressor_victim`: inject `"Aggressor: {victim}; Victim: {aggressor}"` as a guaranteed distractor
  - `compound_aggressor_action_victim`: inject `"{victim} performed {action} on {aggressor}"`
  - `aggressor_identification`: inject the victim as a guaranteed distractor
  - `victim_recognition`: inject the aggressor as a guaranteed distractor
- [ ] Rerun `generate_questions_local.py`
- [ ] Rerun `analyze_distractor_quality.py` and confirm role reversals go from ~1,949 → ~6,000+

**Why first:** 98.2% of videos have both aggressor and victim, so eligibility is essentially universal. This is a one-file change that 3.2x's the most important hard-negative category. Zero data collection required.

**Impact estimate:** role-reversal pairs 1,949 → ~6,200.

---

### P0 — Add wrong-action-same-roles distractors for `compound_aggressor_action_victim`

- [ ] In the same generator pass, for every `compound_aggressor_action_victim` question, inject 2 distractors where the aggressor and victim are kept but the action is swapped to a plausible confusion (punch↔slap, push↔shove, grab↔restrain, tackle↔bodyslam). The action confusion table can be hand-written — there are fewer than 20 aggression actions in the taxonomy.
- [ ] Rerun analysis script to confirm wrong-action count grows by ~2,400 for this category.

**Why:** free signal — the action pool already exists, and 100% of videos have an action field.

**Impact estimate:** +~2,400 hard negatives in `compound_aggressor_action_victim`.

---

### P1 — Restrict identification-question distractor pool to in-cast candidates

- [ ] For `aggressor_identification` and `victim_recognition`, modify the generator to draw distractors from the current video's own annotation first (victim, bystanders, any additional annotated people) before falling back to the global `AnswerBank.people` pool.
- [ ] Keep the guaranteed role-reversal distractor (from P0) as slot 1.
- [ ] Only use global-pool fillers when the local pool is exhausted. When forced to use global, prefer candidates with similar descriptions (colors, clothing tokens) — cheap string-overlap ranking is enough, no embedding model needed.

**Why:** 77% of videos have exactly 2 named people, so this change has a smaller effect than role-reversal injection — the local pool runs out fast. But the 22% of videos with 3+ people (593 videos) see immediate benefit, and the fallback ranking upgrades the remaining filler from "trivially wrong" to "plausibly wrong".

**Impact estimate:** +500-800 hard negatives in the two identification categories. More importantly, fewer trivially rejectable pairs in those categories.

---

### P1 — Collect named bystanders on videos that currently have empty or vague bystander fields

- [ ] Audit `annotations.json`: **1,226 videos (45.6%) have no bystander at all**, and **961 videos (35.8%) have only a vague `"a group of people"` bystander** — useless for substitution.
- [ ] Target: annotate named bystanders on as many of these ~2,200 videos as practical. Even partial coverage helps.
- [ ] Priority: videos where the bystander is visually distinguishable and could plausibly be mistaken for the aggressor (same general appearance as the actual aggressor). These are the highest-value hard negatives.
- [ ] After annotation, enable bystander substitution in the generator:
  - `compound_aggressor_victim`: inject `"Aggressor: {bystander}; Victim: {victim}"` and `"Aggressor: {aggressor}; Victim: {bystander}"`
  - `compound_aggressor_action_victim`: analogous 3-part substitutions
  - `aggressor_identification` / `victim_recognition`: inject bystanders as in-cast distractors

**Why:** this is the single largest *untapped* lever. Bringing the named-bystander rate from 18.6% → 60% would roughly triple bystander-substitution pairs and meaningfully improve identification-question hardness (more videos would have 3+ named people, unlocking the P1 local-pool fix).

**Estimate:** ~12 hours of annotation work at 30 seconds per video.

**Impact estimate (if completed):** +3,000-5,000 additional hard negatives, distributed across all four categories. Also upgrades the P1 identification-pool fix from marginal to meaningful.

---

### P2 — Revise Plan 2's preference-pair target

- [ ] Update `implementation_plans.md` Plan 2 Phase 4 target from "~40,000-60,000 preference pairs" to a dataset-honest number:
  - Realistic ceiling with P0 + P1 fixes and no new annotations: **~13-14K hard negatives across the four focus categories**
  - Realistic ceiling with P1 bystander annotation pass completed: **~18-20K hard negatives**
  - Adding `compound_action_victims` (already 68.9% hard) and partial-cross-video salvage: **~25-30K total preference pairs**
- [ ] Update Plan 2 Phase 4 to explicitly describe role-reversal oversampling (2-3x weight) and the subtype-aware filtering pipeline.
- [ ] Note in the plan that ADPO papers report meaningful gains at 10-30K pairs, so the revised target is still viable.

**Why:** the current 40-60K target implies a preference-pair budget the dataset cannot supply. Leaving it in the plan risks over-promising in a paper draft or grant document.

---

### P2 — Add a `generate_preference_pairs` mode to the generator

- [ ] New output file parallel to `generated_questions_*.json` that contains ADPO-ready preference pairs with structure:
  ```
  {
    "prompt": "...",
    "chosen": "...",
    "rejected": "...",
    "hardness": "role_reversal" | "wrong_action_same_roles" | "bystander_substitution" | "partial_cross_video",
    "video_name": "...",
    "question_type": "..."
  }
  ```
- [ ] Filter out easy cross-video pairs entirely from this file.
- [ ] This separates *benchmark evaluation* (keep the current 8-option MCQ format, including some easy distractors for discrimination) from *training preference data* (only hard pairs, subtype-tagged for loss weighting).

**Why:** ADPO training consumes preference pairs, not MCQ questions. Generating them directly from the generator avoids a fragile post-hoc parsing step in the training pipeline.

---

## Decision checkpoint before Plan 2 training starts

After P0 + P1 generator changes are merged and the analysis script confirms the new hard-negative counts, decide:

1. **Proceed with Plan 2 at revised target (~20-25K pairs)** — fastest path, uses only the bystander annotations that already exist.
2. **Pause Plan 2 for ~2 weeks to complete the bystander annotation pass**, then proceed with the larger (~30K) target. Higher ceiling, better role-reversal coverage.

The analysis scripts in this directory should be rerun after any generator or annotation change to measure actual impact against the projections above.
