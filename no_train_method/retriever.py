from __future__ import annotations

import hashlib
import json
import string
from collections import defaultdict


def load_train_data(path: str) -> list[dict]:
    """Load training questions, auto-detecting dict vs list format."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions_by_video" in data:
        flat: list[dict] = []
        for video_questions in data["questions_by_video"].values():
            flat.extend(video_questions)
        return flat
    raise ValueError(
        f"Unrecognized training data format in {path}: "
        f"expected a list or a dict with 'questions_by_video' key"
    )


def build_parent_group_map(dataset: dict) -> dict[str, str]:
    """Map each video clip to its parent group ID.

    Grouped clips share the parent key as their group ID.
    Solo clips (under "no_group") each get a unique solo ID.
    """
    video_to_group: dict[str, str] = {}
    for category, groups in dataset.items():
        for parent_key, clips in groups.items():
            for clip in clips:
                if parent_key == "no_group":
                    group_id = f"{category}/__solo__/{clip}"
                else:
                    group_id = f"{category}/{parent_key}"
                video_to_group[clip] = group_id
    return video_to_group


def build_retrieval_index(
    train_examples: list[dict],
) -> dict[tuple[str, bool], list[dict]]:
    """Build lookup from (question_type, is_trick) to list of examples."""
    index: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for ex in train_examples:
        key = (ex["question_type"], ex["is_trick"])
        index[key].append(ex)
    return dict(index)


def retrieve(
    test_example: dict,
    index: dict[tuple[str, bool], list[dict]],
    video_to_group: dict[str, str],
) -> dict | None:
    """Find a training example matching the test instance.

    Matches on (question_type, is_trick), falling back to the opposite
    is_trick value. Excludes candidates from the same parent video group.
    Selection is deterministic via SHA-256 seeding (not Python hash()).
    """
    qtype = test_example["question_type"]
    trick = test_example["is_trick"]

    candidates = index.get((qtype, trick))
    if candidates is None:
        candidates = index.get((qtype, not trick))
    if candidates is None:
        return None

    test_group = video_to_group.get(test_example["video_name"])
    filtered = [
        c for c in candidates
        if video_to_group.get(c["video_name"]) != test_group
    ]
    if not filtered:
        return None

    # Deterministic selection: sort by video_name + prompt, then pick using
    # a stable hash derived from the test example (avoids Python's randomized
    # hash()).
    filtered.sort(key=lambda c: (c["video_name"], c["prompt"]))
    seed_str = f"{test_example['video_name']}|{test_example['prompt']}|{qtype}"
    digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(filtered)
    return filtered[idx]


def _get_answers(example: dict) -> list[str]:
    """Return the answer list, handling both 'answers' and 'all_answers' keys."""
    return example.get("answers") or example.get("all_answers") or []


def _format_options(answers: list[str]) -> str:
    """Format answers as labeled options (A, B, C, ...)."""
    labels = string.ascii_uppercase
    lines = []
    for i, ans in enumerate(answers):
        lines.append(f"{labels[i]}) {ans}")
    return "\n".join(lines)


def build_rap_prompt(
    test_example: dict,
    reference: dict | None,
    n_frames: int,
) -> str:
    """Build the 1-shot retrieval-augmented prompt.

    Layout:
      1. Frame placeholders
      2. Reference example (if provided): question, options, correct answer
      3. Test question with options
    """
    parts: list[str] = []

    # Frame placeholders
    frame_lines = [f"Frame {i + 1}: <image>" for i in range(n_frames)]
    parts.append("\n".join(frame_lines))

    if reference is not None:
        ref_answers = _get_answers(reference)
        correct_label = string.ascii_uppercase[reference["correct_index"]]
        ref_section = (
            "Reference Example:\n"
            f"Question: {reference['prompt']}\n"
            f"{_format_options(ref_answers)}\n"
            f"Correct Answer: {correct_label}"
        )
        parts.append(ref_section)

    # Test question
    test_answers = _get_answers(test_example)
    test_section = (
        f"Question: {test_example['prompt']}\n"
        f"{_format_options(test_answers)}"
    )
    parts.append(test_section)

    return "\n\n".join(parts)
