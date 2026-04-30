#!/usr/bin/env python3
"""
Analyze text-only property frequency bias in the benchmark.

For each question, extracts the properties (aggressor descriptions, victim
descriptions, action verbs, environments) from the answer options and checks
whether the correct answer can be predicted from text frequency alone —
without seeing the video.

If "person in red shirt" appears as the correct answer more often than as a
distractor across the benchmark, a text-only model could exploit that bias.

Also checks whether the top models' answer distributions correlate with
property frequency (i.e., are they picking the most frequently-correct
property rather than interpreting the video?).

Usage:
    python analysis_scripts/text_property_frequency.py \
        --questions train_model/data/generated_questions.json \
        --results-dir combined_results/ \
        --top-n 5
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MODEL_FILE_MAP = {
    "InternVL2.5-78B-AWQ": "InternVL2.5-78B-AWQ_combined.json",
    "InternVL2.5-8B": "InternVL2.5-8B_combined.json",
    "Qwen3-VL-8B": "qwen3_8B_combined.json",
    "InternVL3.5-8B-DoT": "internvl3_5_dot_combined.json",
    "Ovis2.5-9B-Thinking": "Ovis2.5-9B-Thinking_combined.json",
    "InternVL3-9B": "InternVL3-9B_combined.json",
    "InternVL3.5-8B": "internvl3_5_combined.json",
    "Ovis2.5-9B": "Ovis2.5-9B_combined.json",
    "Qwen3-VL-8B-Thinking": "qwen3_8B_thinking_combined.json",
    "gemma-4-26B": "gemma_combined.json",
    "Qwen2.5-VL-7B": "qwen2_5_7B_combined.json",
    "VideoLLaMA3-7B": "VideoLLaMA3-7B_combined.json",
    "InternVideo2.5-8B": "InternVideo2_5_Chat_8B_combined.json",
    "LLaVA-Video-7B": "LLaVA-Video-7B-Qwen2_combined.json",
    "InternVL3.5-8B-CoT": "internvl3_5_cot_combined.json",
    "Qwen2.5-VL-72B": "qwen2_5_72B_combined.json",
}


def extract_properties(answer_text: str, question_type: str) -> list[str]:
    """Extract person/action/location properties from an answer option."""
    text = answer_text.strip().lower()

    if question_type == "primary_action":
        return [text]

    if question_type == "scene_location":
        return [text]

    if question_type == "role_identification":
        return [text]

    person_patterns = re.findall(
        r"person (?:in |wearing |with )[\w\s,]+?(?=\s+(?:performed|performs|committed|who|;|$))",
        text
    )
    action_patterns = re.findall(
        r"(?:performed|performs|committed|action of)\s+([\w\s]+?)(?:\s+(?:on|against|to)\b|$)",
        text
    )
    properties = person_patterns + action_patterns

    if not properties:
        return [text[:60]]

    return [p.strip() for p in properties if p.strip()]


def analyze_frequency_bias(questions_path: str) -> dict:
    """Analyze how often each property appears as correct vs distractor."""
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    property_as_correct = Counter()
    property_as_distractor = Counter()
    correct_position_counts = Counter()
    total_questions = 0

    for video_name, questions in data["questions_by_video"].items():
        for q in questions:
            if q.get("is_secondary"):
                continue
            total_questions += 1
            answers = q.get("answers", [])
            correct_idx = q.get("correct_index", -1)
            qtype = q["question_type"]

            correct_position_counts[correct_idx] += 1

            for i, ans in enumerate(answers):
                props = extract_properties(ans, qtype)
                if i == correct_idx:
                    for p in props:
                        property_as_correct[p] += 1
                else:
                    for p in props:
                        property_as_distractor[p] += 1

    all_properties = set(property_as_correct.keys()) | set(property_as_distractor.keys())
    bias_scores = {}
    for p in all_properties:
        c = property_as_correct.get(p, 0)
        d = property_as_distractor.get(p, 0)
        total = c + d
        if total >= 10:
            bias_scores[p] = {
                "correct_count": c,
                "distractor_count": d,
                "total": total,
                "correct_rate": c / total,
            }

    return {
        "total_questions": total_questions,
        "correct_position_distribution": dict(correct_position_counts),
        "bias_scores": bias_scores,
    }


def majority_vote_baseline(questions_path: str) -> float:
    """Compute accuracy of always picking the most frequent answer text."""
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)

    answer_freq = Counter()
    total = 0
    correct_by_freq = 0

    for video_name, questions in data["questions_by_video"].items():
        for q in questions:
            if q.get("is_secondary"):
                continue
            answers = q.get("answers", [])
            correct_idx = q.get("correct_index", -1)
            total += 1
            for ans in answers:
                answer_freq[ans.strip().lower()] += 1

    correct_answer_counts = Counter()
    for video_name, questions in data["questions_by_video"].items():
        for q in questions:
            if q.get("is_secondary"):
                continue
            correct_answer_counts[q["correct_answer"].strip().lower()] += 1

    most_common_correct = correct_answer_counts.most_common(20)
    return {
        "total_questions": total,
        "top_correct_answers": most_common_correct,
    }


def analyze_model_vs_frequency(results_path: str, questions_path: str) -> dict:
    """Check if a model's choices correlate with answer text frequency."""
    with open(questions_path, encoding="utf-8") as f:
        qdata = json.load(f)
    with open(results_path, encoding="utf-8") as f:
        rdata = json.load(f)

    answer_global_freq = Counter()
    for video_name, questions in qdata["questions_by_video"].items():
        for q in questions:
            if q.get("is_secondary"):
                continue
            for ans in q.get("answers", []):
                answer_global_freq[ans.strip().lower()] += 1

    chose_most_frequent = 0
    chose_least_frequent = 0
    total_answered = 0

    for r in rdata["results"]:
        if r.get("error") or r.get("model_selected_index") is None:
            continue
        answers = r.get("answers", [])
        if not answers:
            continue
        total_answered += 1
        selected = r["model_selected_index"]

        freqs = [answer_global_freq.get(a.strip().lower(), 0) for a in answers]
        max_freq = max(freqs)
        min_freq = min(freqs)

        if selected < len(freqs):
            if freqs[selected] == max_freq:
                chose_most_frequent += 1
            if freqs[selected] == min_freq:
                chose_least_frequent += 1

    return {
        "total_answered": total_answered,
        "chose_most_frequent": chose_most_frequent,
        "chose_most_frequent_pct": chose_most_frequent / max(1, total_answered) * 100,
        "chose_least_frequent": chose_least_frequent,
        "chose_least_frequent_pct": chose_least_frequent / max(1, total_answered) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="train_model/data/generated_questions.json")
    parser.add_argument("--results-dir", default="combined_results/")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("-o", "--output-dir", default=None, help="Directory for CSV outputs")
    args = parser.parse_args()

    print("=" * 70)
    print("TEXT-ONLY PROPERTY FREQUENCY ANALYSIS")
    print("=" * 70)

    # 1. Correct answer position distribution
    bias = analyze_frequency_bias(args.questions)
    print(f"\nTotal questions analyzed: {bias['total_questions']}")
    print(f"\nCorrect answer position distribution:")
    pos_dist = bias["correct_position_distribution"]
    for idx in sorted(pos_dist.keys()):
        letter = chr(65 + idx) if isinstance(idx, int) and idx >= 0 else "?"
        pct = pos_dist[idx] / bias["total_questions"] * 100
        print(f"  {letter}: {pos_dist[idx]:5d} ({pct:.1f}%)")

    chance = 100.0 / max(len(pos_dist), 1)
    print(f"  Chance level: {chance:.1f}%")

    # 2. Most biased properties
    print(f"\nTop 20 most biased properties (appear as correct more often than expected):")
    print(f"  {'Property':<50} {'Correct':>7} {'Distractor':>10} {'Correct%':>8}")
    print(f"  {'-'*50} {'-'*7} {'-'*10} {'-'*8}")
    sorted_bias = sorted(bias["bias_scores"].items(),
                         key=lambda x: -x[1]["correct_rate"])
    for prop, scores in sorted_bias[:20]:
        display = prop[:50]
        print(f"  {display:<50} {scores['correct_count']:>7} {scores['distractor_count']:>10} {scores['correct_rate']*100:>7.1f}%")

    print(f"\nBottom 20 (rarely correct — strong distractor properties):")
    for prop, scores in sorted_bias[-20:]:
        display = prop[:50]
        print(f"  {display:<50} {scores['correct_count']:>7} {scores['distractor_count']:>10} {scores['correct_rate']*100:>7.1f}%")

    # 3. Majority vote baseline
    mv = majority_vote_baseline(args.questions)
    print(f"\nMost frequent correct answer texts:")
    for ans, count in mv["top_correct_answers"][:15]:
        pct = count / mv["total_questions"] * 100
        print(f"  {count:4d} ({pct:4.1f}%)  {ans[:70]}")

    # 4. Model frequency correlation
    results_dir = Path(args.results_dir)
    overall_accuracies = []
    for name, filename in MODEL_FILE_MAP.items():
        path = results_dir / filename
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        correct = sum(1 for r in data["results"] if r.get("is_correct"))
        total = sum(1 for r in data["results"] if not r.get("error"))
        overall_accuracies.append((name, correct / max(1, total) * 100, filename))

    overall_accuracies.sort(key=lambda x: -x[1])
    top_models = overall_accuracies[:args.top_n]

    print(f"\n{'='*70}")
    print(f"MODEL vs TEXT FREQUENCY CORRELATION (top {args.top_n})")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Accuracy':>8} {'Chose Most Freq':>16} {'Chose Least Freq':>17}")
    print(f"  {'-'*25} {'-'*8} {'-'*16} {'-'*17}")

    for name, acc, filename in top_models:
        path = results_dir / filename
        freq_result = analyze_model_vs_frequency(str(path), args.questions)
        print(f"  {name:<25} {acc:>7.1f}% {freq_result['chose_most_frequent_pct']:>15.1f}% {freq_result['chose_least_frequent_pct']:>16.1f}%")

    print(f"\n  If models rely on text frequency, 'Chose Most Freq' would be high.")
    print(f"  If models use visual information, the distribution should be closer to uniform.")

    if args.output_dir:
        import csv
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "position_distribution.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["position", "count", "percentage"])
            for idx in sorted(pos_dist.keys()):
                letter = chr(65 + idx) if isinstance(idx, int) and idx >= 0 else "?"
                w.writerow([letter, pos_dist[idx], f"{pos_dist[idx]/bias['total_questions']*100:.1f}"])

        with open(out / "property_bias.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["property", "correct_count", "distractor_count", "total", "correct_rate"])
            for prop, scores in sorted_bias:
                w.writerow([prop, scores["correct_count"], scores["distractor_count"],
                           scores["total"], f"{scores['correct_rate']*100:.1f}"])

        with open(out / "model_frequency_correlation.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "accuracy", "chose_most_frequent_pct", "chose_least_frequent_pct"])
            for name, acc, filename in top_models:
                path = results_dir / filename
                freq_result = analyze_model_vs_frequency(str(path), args.questions)
                w.writerow([name, f"{acc:.1f}", f"{freq_result['chose_most_frequent_pct']:.1f}",
                           f"{freq_result['chose_least_frequent_pct']:.1f}"])

        print(f"\nWrote CSVs to {out}/")


if __name__ == "__main__":
    main()
