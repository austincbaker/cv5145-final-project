#!/usr/bin/env python3
"""
Grade free-form model responses using an LLM judge.

Takes model responses + ground-truth annotations and asks a judge LLM to
score each response on multiple dimensions. Outputs per-response grades
and aggregate statistics.

Usage (with Anthropic API):
    python freeform_eval/grade_responses.py \
        --responses freeform_eval/responses_InternVL2.5-8B.json \
        --judge anthropic \
        --judge-model claude-sonnet-4-20250514 \
        -o freeform_eval/grades_InternVL2.5-8B.json

Usage (with OpenAI API):
    python freeform_eval/grade_responses.py \
        --responses freeform_eval/responses_InternVL2.5-8B.json \
        --judge openai \
        --judge-model gpt-4o \
        -o freeform_eval/grades_InternVL2.5-8B.json
"""
import argparse
import json
import os
import time
from pathlib import Path


GRADING_PROMPT = """\
You are grading a vision-language model's response about a video clip.
The model was shown 8 frames from the video and asked:

"{prompt}"

The model responded:
"{response}"

Here is the ground truth about this video:
- Aggressive behavior present: {is_aggressive}
- Action: {action}
- Aggressor: {aggressor}
- Victim: {victim}
- Environment: {environment}
- Bystanders: {bystanders}

Grade the model's response on the following criteria. For each, assign a
score from 0-2 (0 = wrong/missing, 1 = partially correct, 2 = correct):

1. AGGRESSION_DETECTION: Did the model correctly identify whether
   aggressive/socially inappropriate behavior is present?
2. ACTION_IDENTIFICATION: Did the model correctly identify the aggressive
   action (or correctly state no action for non-aggressive clips)?
3. AGGRESSOR_IDENTIFICATION: Did the model correctly identify who performed
   the aggressive action (appearance-based, not by name)?
4. VICTIM_IDENTIFICATION: Did the model correctly identify the target of
   the aggression?
5. REASONING_QUALITY: Is the model's defense/reasoning coherent, specific
   to the video content, and well-structured?

Respond ONLY with a JSON object in this exact format:
{{
  "aggression_detection": <0-2>,
  "action_identification": <0-2>,
  "aggressor_identification": <0-2>,
  "victim_identification": <0-2>,
  "reasoning_quality": <0-2>,
  "brief_justification": "<1-2 sentences explaining your grades>"
}}"""


def grade_with_anthropic(prompt: str, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)


def grade_with_openai(prompt: str, model: str) -> dict:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def build_grading_prompt(entry: dict) -> str:
    gt = entry["ground_truth"]
    return GRADING_PROMPT.format(
        prompt=entry["prompt"],
        response=entry["response"],
        is_aggressive="Yes" if gt["is_aggressive"] else "No",
        action=gt.get("action") or "None (non-aggressive clip)",
        aggressor=gt.get("aggressor") or "None",
        victim=gt.get("victim") or "None",
        environment=gt.get("environment") or "Not annotated",
        bystanders=gt.get("bystanders") or "None annotated",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--judge", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--rate-limit-delay", type=float, default=0.5,
                        help="Seconds between API calls")
    args = parser.parse_args()

    with open(args.responses, encoding="utf-8") as f:
        data = json.load(f)
    responses = data["responses"]

    grade_fn = grade_with_anthropic if args.judge == "anthropic" else grade_with_openai

    grades = []
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            grades = json.load(f)
        start_idx = len(grades)
        print(f"Resuming from checkpoint: {start_idx}/{len(responses)}")

    criteria = ["aggression_detection", "action_identification",
                "aggressor_identification", "victim_identification",
                "reasoning_quality"]

    for i, entry in enumerate(responses[start_idx:], start=start_idx):
        if not entry.get("response"):
            grades.append({
                "video_name": entry["video_name"],
                "scores": None,
                "error": entry.get("error", "no response"),
            })
            continue

        prompt = build_grading_prompt(entry)
        try:
            result = grade_fn(prompt, args.judge_model)
            grades.append({
                "video_name": entry["video_name"],
                "scores": {c: result.get(c, 0) for c in criteria},
                "justification": result.get("brief_justification", ""),
                "is_aggressive": entry["ground_truth"]["is_aggressive"],
            })
            total = sum(result.get(c, 0) for c in criteria)
            print(f"  [{i+1}/{len(responses)}] {entry['video_name']}: {total}/10 - {result.get('brief_justification', '')[:60]}")
        except Exception as e:
            print(f"  [{i+1}/{len(responses)}] ERROR grading {entry['video_name']}: {e}")
            grades.append({
                "video_name": entry["video_name"],
                "scores": None,
                "error": str(e),
            })

        if (i + 1) % 10 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(grades, f, indent=2, ensure_ascii=False)

        time.sleep(args.rate_limit_delay)

    graded = [g for g in grades if g.get("scores")]
    agg_graded = [g for g in graded if g.get("is_aggressive")]
    non_agg_graded = [g for g in graded if not g.get("is_aggressive")]

    def avg_scores(items):
        if not items:
            return {}
        return {
            c: round(sum(g["scores"][c] for g in items) / len(items), 2)
            for c in criteria
        }

    output = {
        "metadata": {
            "model": data["metadata"]["model"],
            "judge": args.judge,
            "judge_model": args.judge_model,
            "num_graded": len(graded),
            "num_errors": len(grades) - len(graded),
        },
        "aggregate": {
            "overall": avg_scores(graded),
            "aggressive_clips": avg_scores(agg_graded),
            "non_aggressive_clips": avg_scores(non_agg_graded),
            "overall_mean": round(sum(sum(g["scores"].values()) for g in graded) / len(graded), 2) if graded else 0,
        },
        "grades": grades,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n{'='*60}")
    print(f"Model: {data['metadata']['model']}")
    print(f"Judge: {args.judge_model}")
    print(f"Graded: {len(graded)}/{len(grades)}")
    print(f"\nOverall scores (0-2 scale):")
    for c, v in output["aggregate"]["overall"].items():
        print(f"  {c:30s}: {v:.2f}")
    print(f"  {'TOTAL':30s}: {output['aggregate']['overall_mean']:.2f}/10")
    if agg_graded:
        print(f"\nAggressive clips ({len(agg_graded)}):")
        for c, v in output["aggregate"]["aggressive_clips"].items():
            print(f"  {c:30s}: {v:.2f}")
    if non_agg_graded:
        print(f"\nNon-aggressive clips ({len(non_agg_graded)}):")
        for c, v in output["aggregate"]["non_aggressive_clips"].items():
            print(f"  {c:30s}: {v:.2f}")
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
