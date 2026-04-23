#!/usr/bin/env python3
"""Phase 4: extract DPO preference pairs.

For each train-split question, build a (chosen, rejected[]) preference pair
where `chosen` is the correct answer (optionally prefixed with CoT reasoning)
and `rejected[]` are the hardest distractors from that question's multiple-
choice options.

Distractor hardness is determined by a structural classifier: the classifier
parses the distractor string against the same templates the question generator
uses, and compares the parsed (aggressor, action, victim, location) slots
against this video's annotation to label:

    role_reversal        — aggressor and victim swapped
    wrong_action         — same cast, different verb
    wrong_victim         — same aggressor + action, different victim
    wrong_aggressor      — correct victim + action, different aggressor
    bystander_substitution — one slot filled by a known bystander
    wrong_location       — correct aggressor, different location
    wrong_category       — fixed-vocabulary contrast (action/location name)
    none_claim           — "No one fits", "No aggressive action", etc.
    other_in_cast        — mentions a cast member but doesn't fit above
    cross_video          — unrelated to this video's annotation

Distractors are sorted by priority (role_reversal hardest, cross_video
easiest) and the top `--num-rejected` (default 5) are kept. Nothing is
pre-filtered, so a question with only cross-video distractors still yields
a usable pair.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


from prompt_generator.hardness import classify_distractor, HARDNESS_PRIORITY

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _load_annotations(annotations_path: str) -> dict[str, dict]:
    with open(annotations_path, encoding="utf-8") as f:
        rows = json.load(f)
    out: dict[str, dict] = {}
    for r in rows:
        key = r.get("file_name") or r.get("video_name")
        if key:
            out[key] = r
    return out


def extract_preference_pairs(
    questions_path: str = "train_model/data/generated_questions.json",
    cot_chains_path: str = "train_model/data/cot_chains.json",
    output_path: str = "train_model/data/preference_pairs.json",
    train_split_path: str = "train_model/data/sft_train.json",
    annotations_path: str = "annotations.json",
    num_rejected_per_chosen: int = 5,
    seed: int = 42,
):
    import random
    random.seed(seed)

    with open(questions_path, encoding="utf-8") as f:
        questions_data = json.load(f)
    questions_by_video = questions_data["questions_by_video"]

    # Train-split filter
    if Path(train_split_path).exists():
        with open(train_split_path, encoding="utf-8") as f:
            train_examples = json.load(f)
        train_videos = {ex["video_name"] for ex in train_examples}
        before = len(questions_by_video)
        questions_by_video = {v: qs for v, qs in questions_by_video.items()
                              if v in train_videos}
        print(f"Train-split filter: kept {len(questions_by_video)}/{before} videos")
    else:
        print(f"WARNING: {train_split_path} not found; no train/val/test split guard.")

    annotations = _load_annotations(annotations_path)
    print(f"Loaded {len(annotations)} annotation records")

    # Optional CoT enrichment
    cot_by_video_qidx: dict = {}
    if Path(cot_chains_path).exists():
        with open(cot_chains_path, encoding="utf-8") as f:
            cot_examples = json.load(f)
        for ex in cot_examples:
            key = (ex.get("video_name"), ex.get("question_index", 0))
            if ex.get("used_cot") and "reasoning_chain" in ex:
                cot_by_video_qidx[key] = ex["reasoning_chain"]
        print(f"Loaded {len(cot_by_video_qidx)} CoT chains")
    else:
        print("No CoT chains found; preference pairs will use direct answers")

    preference_pairs = []
    pair_stats: Counter = Counter()
    skipped_noann = 0

    for video_name, questions in questions_by_video.items():
        ann = annotations.get(video_name, {})
        if not ann:
            skipped_noann += 1
            continue
        for q_idx, q in enumerate(questions):
            if q.get("is_secondary") or q.get("is_trick"):
                continue

            qtype = q["question_type"]
            correct_answer = q["correct_answer"]
            answers = q["answers"]
            correct_index = q.get("correct_index", -1)

            reasoning_chain = cot_by_video_qidx.get((video_name, q_idx))

            # Letter-prefix chosen and rejected answers so DPO trains the same
            # output format SFT produces ("{L}) {text}"). Without this, the
            # SFT checkpoint generates "B) ..." but DPO pushes it toward
            # bare "..." — SFT->DPO format drift that undoes MCQ training.
            # claude_mcq_proposal.md Gap A.
            if correct_index < 0 or correct_index >= len(answers):
                continue  # malformed question; skip
            correct_letter = chr(ord("A") + correct_index)
            chosen = {
                "answer": f"{correct_letter}) {correct_answer}",
                "reasoning": reasoning_chain,
            }

            if q.get("option_hardness"):
                labeled = [
                    (i, a, h) for i, (a, h) in enumerate(zip(answers, q["option_hardness"]))
                    if h != "correct"
                ]
                classified = [(i, a, h) for i, a, h in labeled]
            else:
                classified = []
                for i, a in enumerate(answers):
                    if i == correct_index or a == correct_answer:
                        continue
                    classified.append((i, a, classify_distractor(qtype, a, correct_answer, ann)))

            classified.sort(key=lambda x: HARDNESS_PRIORITY.get(x[2], 99))
            top = classified[:num_rejected_per_chosen]

            # `index` is the distractor's position in `all_answers`. The DPO
            # dataset uses it to re-letter after per-pair option shuffling
            # (claude_mcq_proposal.md Gap B). `answer` is the pre-formatted
            # letter-prefixed string used when shuffling is disabled.
            rejected = [
                {
                    "answer": f"{chr(ord('A') + idx)}) {d}",
                    "hardness": h,
                    "index": idx,
                    "text": d,
                }
                for idx, d, h in top
            ]
            if not rejected:
                continue
            for _, _, h in top:
                pair_stats[h] += 1

            preference_pairs.append({
                "video_name": video_name,
                "question_type": qtype,
                "prompt": q["prompt"],
                "video_context": "",
                "chosen": chosen,
                "rejected": rejected,
                "all_answers": answers,
                "correct_index": correct_index,
            })

    if skipped_noann:
        print(f"Skipped {skipped_noann} videos with no annotation record")

    print(f"\nExtracted {len(preference_pairs)} preference pairs")
    print("Hardness distribution (over all kept rejected responses):")
    for h, c in sorted(pair_stats.items(), key=lambda x: HARDNESS_PRIORITY.get(x[0], 99)):
        print(f"  {h:25s}: {c:6,}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preference_pairs, f, indent=2)
    print(f"\nSaved to {output_path}")

    if preference_pairs:
        avg_rej = sum(len(p["rejected"]) for p in preference_pairs) / len(preference_pairs)
        with_cot = sum(1 for p in preference_pairs if p["chosen"].get("reasoning"))
        print(f"  Total pairs: {len(preference_pairs):,}")
        print(f"  Avg rejected per pair: {avg_rej:.2f}")
        print(f"  With CoT reasoning: {with_cot}")

    return preference_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="train_model/data/generated_questions.json")
    parser.add_argument("--cot-chains", default="train_model/data/cot_chains.json")
    parser.add_argument("--output", default="train_model/data/preference_pairs.json")
    parser.add_argument("--train-split", default="train_model/data/sft_train.json")
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--num-rejected", type=int, default=5)
    args = parser.parse_args()

    extract_preference_pairs(
        questions_path=args.questions,
        cot_chains_path=args.cot_chains,
        output_path=args.output,
        train_split_path=args.train_split,
        annotations_path=args.annotations,
        num_rejected_per_chosen=args.num_rejected,
    )
