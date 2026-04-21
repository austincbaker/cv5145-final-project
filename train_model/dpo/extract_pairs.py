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


# ---------------------------------------------------------------------------
# Template parsers
# ---------------------------------------------------------------------------
# Each returns (aggressor, action, victim, location) with Nones where the
# template does not fill that slot.

RE_AAV = re.compile(
    r"^(?P<agg>.+?)\s+performed\s+(?P<act>.+?)\s+on\s+(?P<vic>.+)$", re.I
)
RE_INTERACTION = re.compile(
    r"^(?P<agg>.+?)\s+performs\s+(?:the\s+)?action\s+of\s+(?P<act>.+?)\s+on\s+(?P<vic>.+)$",
    re.I,
)
RE_SEQ = re.compile(
    r"^(?P<agg>.+?),\s*who\s+is\s+the\s+aggressor,\s*performed\s+the\s+action\s+of\s+"
    r"(?P<act>.+?)\s+against\s+(?P<vic>.+)$",
    re.I,
)
RE_AV = re.compile(
    r"^\s*aggressor:\s*(?P<agg>.+?);\s*victim:\s*(?P<vic>.+?)\s*$", re.I
)
RE_ACT_VICTIM = re.compile(r"^\s*(?P<act>[^;]+?);\s*victim:\s*(?P<vic>.+?)\s*$", re.I)
RE_AGG_LOC = re.compile(
    r"^(?P<agg>.+?)\s+in\s+(?P<loc>.+)$", re.I
)
RE_AGG_LOC_UNCLEAR = re.compile(
    r"^(?P<agg>.+?);\s*location\s+unclear\s*$", re.I
)
RE_SOCIAL = re.compile(
    r"^the\s+action\s+performed\s+by\s+(?P<agg>.+)$", re.I
)


def _parse_by_qtype(qtype: str, text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (agg, act, vic, loc) with Nones for slots this template doesn't fill."""
    t = text.strip()
    if qtype == "compound_aggressor_action_victim":
        m = RE_AAV.match(t)
        if m:
            return m["agg"].strip(), m["act"].strip(), m["vic"].strip(), None
    elif qtype == "interaction_summary":
        m = RE_INTERACTION.match(t)
        if m:
            return m["agg"].strip(), m["act"].strip(), m["vic"].strip(), None
    elif qtype == "sequence_verification":
        m = RE_SEQ.match(t)
        if m:
            return m["agg"].strip(), m["act"].strip(), m["vic"].strip(), None
    elif qtype == "compound_aggressor_victim":
        m = RE_AV.match(t)
        if m:
            return m["agg"].strip(), None, m["vic"].strip(), None
    elif qtype == "compound_action_victims":
        m = RE_ACT_VICTIM.match(t)
        if m:
            return None, m["act"].strip(), m["vic"].strip(), None
    elif qtype == "compound_aggressor_location":
        m = RE_AGG_LOC_UNCLEAR.match(t)
        if m:
            return m["agg"].strip(), None, None, "unclear"
        m = RE_AGG_LOC.match(t)
        if m:
            return m["agg"].strip(), None, None, m["loc"].strip()
    elif qtype == "compound_bystander_location":
        m = RE_AGG_LOC.match(t)
        if m:
            # bystander in slot "agg"
            return None, None, None, m["loc"].strip()
    elif qtype == "social_appropriateness":
        m = RE_SOCIAL.match(t)
        if m:
            return m["agg"].strip(), None, None, None
    return None, None, None, None


# ---------------------------------------------------------------------------
# Fuzzy slot matching
# ---------------------------------------------------------------------------
_STOP = {
    "a", "an", "the", "is", "of", "and", "with", "on", "in", "at", "for",
    "by", "to", "who", "that", "this", "person", "people", "group",
}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", (s or "").lower()) if t not in _STOP}


def _slot_match(a: str | None, b: str | None, thresh: float = 0.5) -> bool:
    """Jaccard-style overlap on content tokens, biased to the smaller side."""
    if not a or not b:
        return False
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / max(1, min(len(ta), len(tb))) >= thresh


def _text_mentions(haystack: str, needle: str | None) -> bool:
    """True if enough of `needle`'s tokens appear in haystack."""
    if not needle:
        return False
    return _slot_match(haystack, needle)


def _bystander_strings(bystanders) -> list[str]:
    """Split annotation['bystanders'] into individual descriptor strings."""
    if not bystanders:
        return []
    if isinstance(bystanders, list):
        return [str(b) for b in bystanders if b]
    # Sometimes comma- or "and"-separated.
    s = str(bystanders)
    parts = re.split(r"\s+and\s+|,\s*", s)
    return [p.strip() for p in parts if p.strip()]


def _any_bystander_match(slot: str | None, bystanders) -> bool:
    if not slot:
        return False
    return any(_slot_match(slot, b) for b in _bystander_strings(bystanders))


# ---------------------------------------------------------------------------
# Hardness classifier
# ---------------------------------------------------------------------------
NONE_MARKERS = (
    "no one", "no individual", "no actions performed", "no aggressive action",
    "no bystanders", "none of the above", "not shown", "unclear",
    "no one appears", "no one in the video",
)


def _is_none_claim(d: str) -> bool:
    dl = d.lower()
    return any(marker in dl for marker in NONE_MARKERS)


def classify_distractor(
    qtype: str,
    distractor: str,
    correct_answer: str,
    annotation: dict,
) -> str:
    agg = str(annotation.get("aggressor") or "").strip()
    vic = str(annotation.get("victim") or "").strip()
    act = str(annotation.get("action") or "").strip()
    bys = annotation.get("bystanders") or ""
    env = str(annotation.get("environment") or "").strip()

    d = distractor.strip()
    dl = d.lower()

    # Universal: negative / abstention distractors
    if _is_none_claim(d):
        return "none_claim"

    # -- role_identification: {Aggressor, Victim, Bystander, "No one..."} --
    if qtype == "role_identification":
        c = correct_answer.lower().strip()
        if dl == "victim" and c == "aggressor":
            return "role_reversal"
        if dl == "aggressor" and c == "victim":
            return "role_reversal"
        if dl == "bystander":
            return "bystander_substitution"
        return "other_in_cast"

    # -- primary_action: fixed vocabulary contrast --
    if qtype == "primary_action":
        return "wrong_category"

    # -- scene_location: fixed vocabulary contrast --
    if qtype == "scene_location":
        return "wrong_category"

    # -- Person-description distractors --
    if qtype in ("aggressor_identification", "perspective_aggressor"):
        if _slot_match(d, vic):
            return "role_reversal"
        if _any_bystander_match(d, bys):
            return "bystander_substitution"
        if _slot_match(d, agg):
            return "other_in_cast"  # shouldn't happen; would equal correct
        return "cross_video"

    if qtype == "victim_recognition":
        if _slot_match(d, agg):
            return "role_reversal"
        if _any_bystander_match(d, bys):
            return "bystander_substitution"
        if _slot_match(d, vic):
            return "other_in_cast"
        return "cross_video"

    if qtype == "bystander_detection":
        if _slot_match(d, agg) or _slot_match(d, vic):
            return "wrong_aggressor"  # participant shown as bystander
        if _any_bystander_match(d, bys):
            return "other_in_cast"  # shouldn't equal correct; similar bystander
        return "cross_video"

    # -- Template-parseable compound distractors --
    d_agg, d_act, d_vic, d_loc = _parse_by_qtype(qtype, d)
    c_agg, c_act, c_vic, c_loc = _parse_by_qtype(qtype, correct_answer)

    # Some templates fill only some slots; fall back to annotation when correct-
    # side slot is unavailable.
    ref_agg = c_agg or agg
    ref_vic = c_vic or vic
    ref_act = c_act or act
    ref_loc = c_loc or env

    # Compound_bystander_location: aggressor slot is actually a bystander.
    if qtype == "compound_bystander_location":
        if d_loc and ref_loc and not _slot_match(d_loc, ref_loc):
            return "wrong_location"
        return "other_in_cast"

    # Aggressor-location family
    if qtype == "compound_aggressor_location":
        if d_agg and _slot_match(d_agg, ref_agg):
            if d_loc and ref_loc and not _slot_match(d_loc, ref_loc):
                return "wrong_location"
            return "other_in_cast"
        if _any_bystander_match(d_agg, bys):
            return "bystander_substitution"
        if _slot_match(d_agg, ref_vic):
            return "role_reversal"
        return "cross_video"

    if qtype == "social_appropriateness":
        if _slot_match(d_agg, ref_vic):
            return "role_reversal"
        if _any_bystander_match(d_agg, bys):
            return "bystander_substitution"
        if _slot_match(d_agg, ref_agg):
            return "other_in_cast"
        return "cross_video"

    # Aggressor + (maybe action) + victim templates
    agg_is_correct_agg = _slot_match(d_agg, ref_agg)
    agg_is_correct_vic = _slot_match(d_agg, ref_vic)
    vic_is_correct_agg = _slot_match(d_vic, ref_agg)
    vic_is_correct_vic = _slot_match(d_vic, ref_vic)
    agg_is_bystander = _any_bystander_match(d_agg, bys)
    vic_is_bystander = _any_bystander_match(d_vic, bys)

    if agg_is_correct_vic and vic_is_correct_agg:
        return "role_reversal"

    if agg_is_correct_agg and vic_is_correct_vic:
        # Same cast — action must be wrong (otherwise this would equal correct)
        if d_act and ref_act and not _slot_match(d_act, ref_act):
            return "wrong_action"
        return "other_in_cast"

    if agg_is_correct_agg and not vic_is_correct_vic:
        return "bystander_substitution" if vic_is_bystander else "wrong_victim"

    if vic_is_correct_vic and not agg_is_correct_agg:
        return "bystander_substitution" if agg_is_bystander else "wrong_aggressor"

    if agg_is_bystander or vic_is_bystander:
        return "bystander_substitution"

    if agg_is_correct_agg or vic_is_correct_vic or agg_is_correct_vic or vic_is_correct_agg:
        return "other_in_cast"

    return "cross_video"


# Priority: lower = harder = higher preference for inclusion.
HARDNESS_PRIORITY = {
    "role_reversal": 0,
    "wrong_action": 1,
    "wrong_victim": 2,
    "wrong_aggressor": 3,
    "bystander_substitution": 4,
    "wrong_location": 5,
    "wrong_category": 6,
    "none_claim": 7,
    "other_in_cast": 8,
    "cross_video": 9,
}


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
            chosen = {"answer": correct_answer, "reasoning": reasoning_chain}

            distractors = [a for i, a in enumerate(answers)
                           if i != correct_index and a != correct_answer]
            classified = [(d, classify_distractor(qtype, d, correct_answer, ann))
                          for d in distractors]
            classified.sort(key=lambda x: HARDNESS_PRIORITY.get(x[1], 99))
            top = classified[:num_rejected_per_chosen]

            rejected = [{"answer": d, "hardness": h} for d, h in top]
            if not rejected:
                continue
            for _, h in top:
                pair_stats[h] += 1

            preference_pairs.append({
                "video_name": video_name,
                "question_type": qtype,
                "prompt": q["prompt"],
                "video_context": "",
                "chosen": chosen,
                "rejected": rejected,
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
