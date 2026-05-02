"""Tests for select_rejected_adaptive in train_model/dpo/extract_pairs.py."""

import json

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


class TestExtractPairsAdaptive:
    """Integration: extract_preference_pairs with adaptive selection."""

    def _write_json(self, path, data):
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _make_fixtures(self, tmp_path):
        q1_prompt = "What action is happening?"
        q2_prompt = "Who is the aggressor?"
        questions = {
            "questions_by_video": {
                "video1.mp4": [
                    {
                        "video_name": "video1.mp4",
                        "question_type": "primary_action",
                        "prompt": q1_prompt,
                        "correct_answer": "punch",
                        "answers": ["punch", "kick", "shove", "slap",
                                    "push", "grab", "trip", "none"],
                        "correct_index": 0,
                        "option_hardness": ["correct", "wrong_category",
                                            "wrong_category", "wrong_category",
                                            "wrong_category", "wrong_category",
                                            "cross_video", "none_claim"],
                    },
                    {
                        "video_name": "video1.mp4",
                        "question_type": "aggressor_identification",
                        "prompt": q2_prompt,
                        "correct_answer": "person in red",
                        "answers": ["person in red", "person in blue",
                                    "person in green", "person in black",
                                    "person in white", "unknown person",
                                    "bystander A", "no one"],
                        "correct_index": 0,
                        "option_hardness": ["correct", "role_reversal",
                                            "wrong_aggressor",
                                            "bystander_substitution",
                                            "cross_video", "cross_video",
                                            "other_in_cast", "none_claim"],
                    },
                ],
            },
        }
        eval_results = {
            "results": [
                {"video_name": "video1.mp4", "prompt": q1_prompt, "is_correct": True},
                {"video_name": "video1.mp4", "prompt": q2_prompt, "is_correct": False},
            ],
        }
        annotations = [{"file_name": "video1.mp4", "action": "punch",
                         "aggressor": "person in red", "victim": "person in blue",
                         "environment": "outdoor", "bystanders": "person in green"}]

        qpath = tmp_path / "questions.json"
        epath = tmp_path / "eval.json"
        apath = tmp_path / "annotations.json"
        opath = tmp_path / "pairs.json"
        for p, d in [(qpath, questions), (epath, eval_results), (apath, annotations)]:
            self._write_json(p, d)
        return str(qpath), str(epath), str(apath), str(opath), q1_prompt, q2_prompt

    def test_hard_mining_correct_gets_easy_wrong_gets_hard(self, tmp_path):
        qpath, epath, apath, opath, q1_prompt, q2_prompt = self._make_fixtures(tmp_path)
        from train_model.dpo.extract_pairs import extract_preference_pairs

        pairs = extract_preference_pairs(
            questions_path=qpath,
            cot_chains_path="/nonexistent",
            output_path=opath,
            train_split_path="/nonexistent",
            annotations_path=apath,
            num_rejected_per_chosen=1,
            eval_results_path=epath,
            selection_strategy="hard_mining",
        )

        assert len(pairs) == 2
        q1_pair = next(p for p in pairs if p["prompt"] == q1_prompt)
        q2_pair = next(p for p in pairs if p["prompt"] == q2_prompt)

        # Q1: model correct -> easiest rejected
        assert q1_pair["rejected"][0]["hardness"] in ("cross_video", "none_claim")

        # Q2: model wrong -> hardest rejected
        assert q2_pair["rejected"][0]["hardness"] == "role_reversal"

    def test_fixed_ignores_eval_results(self, tmp_path):
        qpath, epath, apath, opath, q1_prompt, q2_prompt = self._make_fixtures(tmp_path)
        from train_model.dpo.extract_pairs import extract_preference_pairs

        pairs = extract_preference_pairs(
            questions_path=qpath,
            cot_chains_path="/nonexistent",
            output_path=opath,
            train_split_path="/nonexistent",
            annotations_path=apath,
            num_rejected_per_chosen=1,
            eval_results_path=epath,
            selection_strategy="fixed",
        )

        # Fixed always picks hardest regardless of eval results
        for p in pairs:
            h = p["rejected"][0]["hardness"]
            assert h not in ("cross_video", "none_claim"), \
                f"Fixed strategy should pick hardest, got {h}"

    def test_no_eval_results_falls_back(self, tmp_path):
        qpath, _, apath, opath, q1_prompt, q2_prompt = self._make_fixtures(tmp_path)
        from train_model.dpo.extract_pairs import extract_preference_pairs

        pairs = extract_preference_pairs(
            questions_path=qpath,
            cot_chains_path="/nonexistent",
            output_path=opath,
            train_split_path="/nonexistent",
            annotations_path=apath,
            num_rejected_per_chosen=1,
            eval_results_path=None,
            selection_strategy="hard_mining",
        )

        # No eval results -> falls back to hardest
        for p in pairs:
            h = p["rejected"][0]["hardness"]
            assert h not in ("cross_video", "none_claim"), \
                f"No eval results should fall back to hardest, got {h}"
