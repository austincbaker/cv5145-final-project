from __future__ import annotations

import json

import pytest

from no_train_method.retriever import (
    build_parent_group_map,
    build_retrieval_index,
    build_rap_prompt,
    load_train_data,
    retrieve,
    _get_answers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATASET = {
    "punch": {
        "punch_1": ["punch_1_trim_0.mp4", "punch_1_trim_1.mp4"],
        "punch_2": ["punch_2_trim_0.mp4"],
        "no_group": ["punch_solo_001.mp4"],
    },
    "kick": {
        "kick_1": ["kick_1_trim_0.mp4", "kick_1_trim_1.mp4"],
        "no_group": ["kick_solo_001.mp4"],
    },
}


def _make_example(
    video: str,
    qtype: str = "primary_action",
    trick: bool = False,
    prompt: str = "What action?",
    answers: list[str] | None = None,
    correct_index: int = 0,
    correct_answer: str = "punch",
    use_all_answers_key: bool = False,
) -> dict:
    ans = answers or ["punch", "kick", "slap", "shove"]
    ex = {
        "video_name": video,
        "question_type": qtype,
        "is_trick": trick,
        "prompt": prompt,
        "correct_index": correct_index,
        "correct_answer": correct_answer,
    }
    if use_all_answers_key:
        ex["all_answers"] = ans
    else:
        ex["answers"] = ans
    return ex


# ---------------------------------------------------------------------------
# TestParentGroupMap
# ---------------------------------------------------------------------------

class TestParentGroupMap:
    def test_grouped_clips_share_parent(self):
        mapping = build_parent_group_map(DATASET)
        assert mapping["punch_1_trim_0.mp4"] == mapping["punch_1_trim_1.mp4"]
        assert mapping["punch_1_trim_0.mp4"] == "punch/punch_1"

    def test_no_group_clips_get_solo_id(self):
        mapping = build_parent_group_map(DATASET)
        assert mapping["punch_solo_001.mp4"] == "punch/__solo__/punch_solo_001.mp4"
        assert mapping["kick_solo_001.mp4"] == "kick/__solo__/kick_solo_001.mp4"
        # Solo clips must not share a group ID
        assert mapping["punch_solo_001.mp4"] != mapping["kick_solo_001.mp4"]

    def test_multiple_categories(self):
        mapping = build_parent_group_map(DATASET)
        # Clips from different categories must have different group IDs
        assert mapping["punch_1_trim_0.mp4"] != mapping["kick_1_trim_0.mp4"]
        # All expected clips are present
        assert len(mapping) == 7


# ---------------------------------------------------------------------------
# TestRetrievalIndex
# ---------------------------------------------------------------------------

class TestRetrievalIndex:
    def test_index_keys(self):
        examples = [
            _make_example("v1.mp4", qtype="primary_action", trick=False),
            _make_example("v2.mp4", qtype="primary_action", trick=False),
            _make_example("v3.mp4", qtype="role_identification", trick=True),
        ]
        index = build_retrieval_index(examples)
        assert ("primary_action", False) in index
        assert ("role_identification", True) in index
        assert len(index[("primary_action", False)]) == 2
        assert len(index[("role_identification", True)]) == 1

    def test_index_empty_input(self):
        index = build_retrieval_index([])
        assert index == {}


# ---------------------------------------------------------------------------
# TestRetrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _build_index_and_map(self, train_examples):
        index = build_retrieval_index(train_examples)
        video_to_group = build_parent_group_map(DATASET)
        return index, video_to_group

    def test_retrieves_matching_type(self):
        train = [
            _make_example("kick_1_trim_0.mp4", qtype="primary_action"),
        ]
        index, vtg = self._build_index_and_map(train)
        test_ex = _make_example("punch_1_trim_0.mp4", qtype="primary_action")
        result = retrieve(test_ex, index, vtg)
        assert result is not None
        assert result["question_type"] == "primary_action"

    def test_excludes_same_parent_group(self):
        # Both train examples are from the same parent group as the test clip
        train = [
            _make_example("punch_1_trim_1.mp4", qtype="primary_action"),
            _make_example("kick_1_trim_0.mp4", qtype="primary_action"),
        ]
        index, vtg = self._build_index_and_map(train)
        test_ex = _make_example("punch_1_trim_0.mp4", qtype="primary_action")
        result = retrieve(test_ex, index, vtg)
        assert result is not None
        # Must not return the same-group clip
        assert result["video_name"] == "kick_1_trim_0.mp4"

    def test_returns_none_when_all_same_group(self):
        train = [
            _make_example("punch_1_trim_1.mp4", qtype="primary_action"),
        ]
        index, vtg = self._build_index_and_map(train)
        test_ex = _make_example("punch_1_trim_0.mp4", qtype="primary_action")
        result = retrieve(test_ex, index, vtg)
        assert result is None

    def test_falls_back_to_any_trick_value(self):
        train = [
            _make_example("kick_1_trim_0.mp4", qtype="primary_action", trick=True),
        ]
        index, vtg = self._build_index_and_map(train)
        # Test example has trick=False, but only trick=True candidates exist
        test_ex = _make_example("punch_1_trim_0.mp4", qtype="primary_action", trick=False)
        result = retrieve(test_ex, index, vtg)
        assert result is not None
        assert result["is_trick"] is True

    def test_returns_none_for_unknown_type(self):
        train = [
            _make_example("kick_1_trim_0.mp4", qtype="primary_action"),
        ]
        index, vtg = self._build_index_and_map(train)
        test_ex = _make_example("punch_1_trim_0.mp4", qtype="nonexistent_type")
        result = retrieve(test_ex, index, vtg)
        assert result is None

    def test_deterministic_across_calls(self):
        train = [
            _make_example("kick_1_trim_0.mp4", qtype="primary_action", prompt="Q1"),
            _make_example("kick_1_trim_1.mp4", qtype="primary_action", prompt="Q2"),
            _make_example("punch_2_trim_0.mp4", qtype="primary_action", prompt="Q3"),
        ]
        index, vtg = self._build_index_and_map(train)
        test_ex = _make_example("punch_1_trim_0.mp4", qtype="primary_action")
        results = [retrieve(test_ex, index, vtg) for _ in range(20)]
        # All calls must return the exact same candidate
        assert all(r["video_name"] == results[0]["video_name"] for r in results)
        assert all(r["prompt"] == results[0]["prompt"] for r in results)


# ---------------------------------------------------------------------------
# TestBuildRapPrompt
# ---------------------------------------------------------------------------

class TestBuildRapPrompt:
    def test_with_reference(self):
        ref = _make_example(
            "kick_1_trim_0.mp4",
            prompt="What is the aggressive action?",
            answers=["punch", "kick", "slap"],
            correct_index=1,
            correct_answer="kick",
        )
        test_ex = _make_example(
            "punch_1_trim_0.mp4",
            prompt="What action is shown?",
            answers=["shove", "punch", "elbow"],
            correct_index=1,
        )
        prompt = build_rap_prompt(test_ex, ref, n_frames=3)

        # Frame placeholders
        assert "Frame 1: <image>" in prompt
        assert "Frame 2: <image>" in prompt
        assert "Frame 3: <image>" in prompt

        # Reference section
        assert "Reference Example:" in prompt
        assert "What is the aggressive action?" in prompt
        assert "A. punch" in prompt
        assert "B. kick" in prompt
        assert "C. slap" in prompt
        assert "Correct Answer: B" in prompt

        # Test question section
        assert "What action is shown?" in prompt
        assert "A. shove" in prompt
        assert "B. punch" in prompt

        # Reference must come before the test question
        ref_pos = prompt.index("Reference Example:")
        test_pos = prompt.index("What action is shown?")
        assert ref_pos < test_pos

    def test_without_reference_is_standard_prompt(self):
        test_ex = _make_example(
            "punch_1_trim_0.mp4",
            prompt="What action is shown?",
            answers=["punch", "kick"],
        )
        prompt = build_rap_prompt(test_ex, None, n_frames=2)

        assert "Frame 1: <image>" in prompt
        assert "Frame 2: <image>" in prompt
        assert "What action is shown?" in prompt
        assert "A. punch" in prompt
        assert "B. kick" in prompt
        # No reference text
        assert "Reference" not in prompt
        assert "Correct Answer" not in prompt

    def test_handles_all_answers_key(self):
        test_ex = _make_example(
            "punch_1_trim_0.mp4",
            prompt="Identify the action",
            answers=["punch", "kick", "slap"],
            use_all_answers_key=True,
        )
        ref = _make_example(
            "kick_1_trim_0.mp4",
            prompt="What happened?",
            answers=["kick", "punch"],
            correct_index=0,
            use_all_answers_key=True,
        )
        prompt = build_rap_prompt(test_ex, ref, n_frames=1)

        # Should still render options from all_answers
        assert "A. kick" in prompt
        assert "B. punch" in prompt
        assert "A. punch" in prompt
        assert "B. kick" in prompt
        assert "C. slap" in prompt


# ---------------------------------------------------------------------------
# TestEndToEnd -- integration tests for the full pipeline
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Integration test: full retrieval + prompt pipeline with mock data."""

    def test_full_pipeline_questions_by_video_format(self, tmp_path):
        dataset = {
            "punch": {"p1": ["punch_001.mp4", "punch_002.mp4"]},
            "kick": {"k1": ["kick_001.mp4"]},
        }
        train_data = {
            "metadata": {"total_questions": 2},
            "questions_by_video": {
                "punch_001.mp4": [
                    {
                        "video_name": "punch_001.mp4",
                        "question_type": "aggressor_identification",
                        "is_trick": False,
                        "prompt": "Who is the aggressor?",
                        "answers": ["Alice", "Bob", "Carol"],
                        "correct_answer": "Bob",
                        "correct_index": 1,
                        "option_hardness": ["wrong_aggressor", "correct",
                                            "role_reversal"],
                    }
                ],
                "kick_001.mp4": [
                    {
                        "video_name": "kick_001.mp4",
                        "question_type": "primary_action",
                        "is_trick": False,
                        "prompt": "What action?",
                        "answers": ["punch", "kick"],
                        "correct_answer": "kick",
                        "correct_index": 1,
                        "option_hardness": ["wrong_action", "correct"],
                    }
                ],
            },
        }
        test_examples = [
            {
                "video_name": "kick_001.mp4",
                "question_type": "aggressor_identification",
                "is_trick": False,
                "prompt": "Who started the fight?",
                "answers": ["Dave", "Eve"],
                "correct_answer": "Eve",
                "correct_index": 1,
            }
        ]

        # Save training data in questions_by_video format
        train_path = tmp_path / "train.json"
        train_path.write_text(json.dumps(train_data), encoding="utf-8")

        # Load using the retriever helper
        loaded = load_train_data(str(train_path))
        assert len(loaded) == 2

        # Build retriever
        video_to_group = build_parent_group_map(dataset)
        index = build_retrieval_index(loaded)

        # Retrieve for test example
        ref = retrieve(test_examples[0], index, video_to_group)
        assert ref is not None
        assert ref["video_name"] == "punch_001.mp4"

        # Build prompt
        prompt = build_rap_prompt(test_examples[0], ref, n_frames=2)
        assert "Frame 1: <image>" in prompt
        assert "Frame 2: <image>" in prompt
        assert "Reference" in prompt
        assert "Who is the aggressor?" in prompt
        assert "B. Bob" in prompt
        assert "Correct Answer: B" in prompt
        assert "Who started the fight?" in prompt
        assert "A. Dave" in prompt

    def test_full_pipeline_flat_list_format(self, tmp_path):
        train_data = [
            {
                "video_name": "shove_001.mp4",
                "question_type": "victim_recognition",
                "is_trick": False,
                "prompt": "Who was shoved?",
                "all_answers": ["X", "Y"],
                "correct_answer": "X",
                "correct_index": 0,
            }
        ]
        train_path = tmp_path / "train.json"
        train_path.write_text(json.dumps(train_data), encoding="utf-8")

        loaded = load_train_data(str(train_path))
        assert len(loaded) == 1
        assert loaded[0]["question_type"] == "victim_recognition"
