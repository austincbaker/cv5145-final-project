"""Tests for select_rejected_adaptive in train_model/dpo/extract_pairs.py."""

import pytest

from train_model.dpo.extract_pairs import select_rejected_adaptive


# Helpers
def _make_classified(*hardness_labels):
    """Build classified tuples: (index, answer_text, hardness_label)."""
    return [(i, f"answer_{h}", h) for i, h in enumerate(hardness_labels)]


FULL_SPREAD = _make_classified(
    "role_reversal", "wrong_action", "wrong_victim",
    "wrong_aggressor", "bystander_substitution", "wrong_location",
    "wrong_category", "none_claim", "other_in_cast", "cross_video",
)


class TestHardMining:
    """hard_mining strategy: wrong -> hardest, correct -> easiest."""

    def test_model_wrong_picks_hardest(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=False,
                                          strategy="hard_mining", n=3)
        labels = [r[2] for r in result]
        assert labels == ["role_reversal", "wrong_action", "wrong_victim"]

    def test_model_correct_picks_easiest(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=True,
                                          strategy="hard_mining", n=3)
        labels = [r[2] for r in result]
        assert labels == ["cross_video", "other_in_cast", "none_claim"]

    def test_model_correct_n3(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=True,
                                          strategy="hard_mining", n=3)
        assert len(result) == 3
        labels = [r[2] for r in result]
        assert labels[0] == "cross_video"

    def test_none_falls_back_to_hardest(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=None,
                                          strategy="hard_mining", n=3)
        labels = [r[2] for r in result]
        assert labels == ["role_reversal", "wrong_action", "wrong_victim"]


class TestFixed:
    """fixed strategy: always pick hardest regardless of model correctness."""

    def test_always_hardest_when_correct(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=True,
                                          strategy="fixed", n=3)
        labels = [r[2] for r in result]
        assert labels == ["role_reversal", "wrong_action", "wrong_victim"]

    def test_always_hardest_when_wrong(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=False,
                                          strategy="fixed", n=3)
        labels = [r[2] for r in result]
        assert labels == ["role_reversal", "wrong_action", "wrong_victim"]


class TestCurriculum:
    """curriculum strategy: inverse of hard_mining (ablation)."""

    def test_model_wrong_picks_easiest(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=False,
                                          strategy="curriculum", n=3)
        labels = [r[2] for r in result]
        assert labels == ["cross_video", "other_in_cast", "none_claim"]

    def test_model_correct_picks_hardest(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=True,
                                          strategy="curriculum", n=3)
        labels = [r[2] for r in result]
        assert labels == ["role_reversal", "wrong_action", "wrong_victim"]


class TestEdgeCases:

    def test_empty_classified_returns_empty(self):
        result = select_rejected_adaptive([], model_correct=True,
                                          strategy="hard_mining", n=5)
        assert result == []

    def test_n_larger_than_available(self):
        small = _make_classified("role_reversal", "cross_video")
        result = select_rejected_adaptive(small, model_correct=True,
                                          strategy="hard_mining", n=10)
        assert len(result) == 2

    def test_n_zero_returns_empty(self):
        result = select_rejected_adaptive(FULL_SPREAD, model_correct=True,
                                          strategy="hard_mining", n=0)
        assert result == []
