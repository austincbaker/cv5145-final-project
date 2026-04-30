#!/usr/bin/env python3
"""
Grade social appropriateness responses using gpt-oss-20b as judge.

Takes model responses from run_social_appropriateness.py and sends each
response + ground truth to a judge LLM. The judge determines whether the
model correctly identified if the actions were socially appropriate or not.

Scoring is binary:
  - correct: model correctly identified appropriateness (aggressive=inappropriate,
    non-aggressive=appropriate)
  - incorrect: model said appropriate when actions were actually aggressive,
    or said inappropriate when actions were actually non-aggressive

Supports gpt-oss-20b via vLLM (OpenAI-compatible API) or Ollama, as well
as Gemini/Anthropic/OpenAI cloud APIs as fallback.

Usage (vLLM on cluster):
    # Terminal 1: start vLLM server
    vllm serve openai/gpt-oss-20b --port 8000

    # Terminal 2: run grading
    python freeform_eval/grade_social_appropriateness.py \
        --responses freeform_eval/social_responses_InternVL2.5-8B.json \
        --judge vllm \
        --judge-url http://localhost:8000/v1 \
        -o freeform_eval/social_grades_InternVL2.5-8B.json

Usage (Ollama local):
    ollama pull gpt-oss:20b
    python freeform_eval/grade_social_appropriateness.py \
        --responses freeform_eval/social_responses_InternVL2.5-8B.json \
        --judge ollama \
        -o freeform_eval/social_grades_InternVL2.5-8B.json

Usage (Gemini fallback):
    python freeform_eval/grade_social_appropriateness.py \
        --responses freeform_eval/social_responses_InternVL2.5-8B.json \
        --judge gemini \
        --judge-model gemini-2.5-flash \
        -o freeform_eval/social_grades_InternVL2.5-8B.json
"""
import argparse
import json
import os
import time
from pathlib import Path


GRADING_PROMPT = """\
You are evaluating whether a vision-language model correctly assessed the social appropriateness of actions in a video.

The model was shown 8 frames from a video and asked:
"How would you characterize the appropriateness of the actions depicted in the video? Defend your reasoning."

The model responded:
"{response}"

Ground truth about this video:
- Aggressive behavior present: {is_aggressive}
- Action: {action}
- Aggressor: {aggressor}
- Victim: {victim}

Your task: Determine whether the model's response correctly identifies the social appropriateness of the actions.

Rules:
- If the video contains aggressive behavior (is_aggressive=Yes) and the model says the actions are inappropriate, aggressive, wrong, harmful, violent, or socially unacceptable -> CORRECT
- If the video contains aggressive behavior (is_aggressive=Yes) and the model says the actions are appropriate, acceptable, normal, or fine -> INCORRECT
- If the video does NOT contain aggressive behavior (is_aggressive=No) and the model says the actions are appropriate, normal, acceptable, or non-aggressive -> CORRECT
- If the video does NOT contain aggressive behavior (is_aggressive=No) and the model says the actions are inappropriate, aggressive, or harmful -> INCORRECT
- If the model's response is ambiguous, unclear, or does not address appropriateness -> INCORRECT

Respond ONLY with a JSON object:
{{
  "is_correct": true or false,
  "model_said_appropriate": true or false,
  "ground_truth_aggressive": true or false,
  "brief_justification": "<1 sentence explaining your judgment>"
}}"""


def grade_with_vllm(prompt: str, model: str, base_url: str) -> dict:
    import openai
    client = openai.OpenAI(base_url=base_url, api_key="none")
    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)


def grade_with_ollama(prompt: str, model: str, base_url: str = None) -> dict:
    import openai
    client = openai.OpenAI(
        base_url=base_url or "http://localhost:11434/v1",
        api_key="ollama",
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)


def grade_with_gemini(prompt: str, model: str, base_url: str = None) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=200,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        ),
    )
    return json.loads(response.text)


def grade_with_openai(prompt: str, model: str, base_url: str = None) -> dict:
    import openai
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def build_grading_prompt(entry: dict) -> str:
    gt = entry["ground_truth"]
    return GRADING_PROMPT.format(
        response=entry["response"],
        is_aggressive="Yes" if gt["is_aggressive"] else "No",
        action=gt.get("action") or "None (non-aggressive clip)",
        aggressor=gt.get("aggressor") or "None",
        victim=gt.get("victim") or "None",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--judge", choices=["vllm", "ollama", "gemini", "openai"], default="vllm")
    parser.add_argument("--judge-model", default="openai/gpt-oss-20b",
                        help="Model name (default: openai/gpt-oss-20b for vllm/ollama, gemini-2.5-flash for gemini)")
    parser.add_argument("--judge-url", default=None,
                        help="API base URL (default: http://localhost:8000/v1 for vllm, http://localhost:11434/v1 for ollama)")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--rate-limit-delay", type=float, default=0.1)
    args = parser.parse_args()

    if args.judge == "vllm" and not args.judge_url:
        args.judge_url = "http://localhost:8000/v1"
    if args.judge == "ollama" and not args.judge_url:
        args.judge_url = "http://localhost:11434/v1"
    if args.judge == "ollama" and args.judge_model == "openai/gpt-oss-20b":
        args.judge_model = "gpt-oss:20b"
    if args.judge == "gemini" and args.judge_model == "openai/gpt-oss-20b":
        args.judge_model = "gemini-2.5-flash"

    grade_fns = {
        "vllm": grade_with_vllm,
        "ollama": grade_with_ollama,
        "gemini": grade_with_gemini,
        "openai": grade_with_openai,
    }
    grade_fn = grade_fns[args.judge]

    with open(args.responses, encoding="utf-8") as f:
        data = json.load(f)
    responses = data["responses"]
    print(f"Loaded {len(responses)} responses from {data['metadata']['model']}")

    grades = []
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            grades = json.load(f)
        start_idx = len(grades)
        print(f"Resuming from checkpoint: {start_idx}/{len(responses)}")

    correct_count = sum(1 for g in grades if g.get("is_correct"))

    for i, entry in enumerate(responses[start_idx:], start=start_idx):
        if not entry.get("response"):
            grades.append({
                "video_name": entry["video_name"],
                "response": None,
                "is_correct": False,
                "error": entry.get("error", "no response"),
                "ground_truth": entry["ground_truth"],
            })
            continue

        prompt = build_grading_prompt(entry)
        try:
            result = grade_fn(prompt, args.judge_model, args.judge_url)
            is_correct = bool(result.get("is_correct", False))
            if is_correct:
                correct_count += 1
            grades.append({
                "video_name": entry["video_name"],
                "prompt": entry["prompt"],
                "response": entry["response"],
                "is_correct": is_correct,
                "model_said_appropriate": result.get("model_said_appropriate"),
                "ground_truth_aggressive": result.get("ground_truth_aggressive"),
                "justification": result.get("brief_justification", ""),
                "ground_truth": entry["ground_truth"],
            })
            status = "CORRECT" if is_correct else "WRONG"
            running_acc = correct_count / (i + 1) * 100
            print(f"  [{i+1}/{len(responses)}] {status} {entry['video_name']}: {result.get('brief_justification', '')[:60]} (acc: {running_acc:.1f}%)")
        except Exception as e:
            print(f"  [{i+1}/{len(responses)}] ERROR {entry['video_name']}: {e}")
            grades.append({
                "video_name": entry["video_name"],
                "prompt": entry["prompt"],
                "response": entry["response"],
                "is_correct": None,
                "error": str(e),
                "ground_truth": entry["ground_truth"],
            })

        if (i + 1) % 10 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(grades, f, indent=2, ensure_ascii=False)

        time.sleep(args.rate_limit_delay)

    graded = [g for g in grades if g.get("is_correct") is not None]
    correct = sum(1 for g in graded if g["is_correct"])
    agg_graded = [g for g in graded if g["ground_truth"]["is_aggressive"]]
    non_agg_graded = [g for g in graded if not g["ground_truth"]["is_aggressive"]]
    agg_correct = sum(1 for g in agg_graded if g["is_correct"])
    non_agg_correct = sum(1 for g in non_agg_graded if g["is_correct"])

    output = {
        "metadata": {
            "model": data["metadata"]["model"],
            "judge": args.judge,
            "judge_model": args.judge_model,
            "num_graded": len(graded),
            "num_errors": len(grades) - len(graded),
        },
        "aggregate": {
            "overall_accuracy": round(correct / max(1, len(graded)) * 100, 2),
            "aggressive_accuracy": round(agg_correct / max(1, len(agg_graded)) * 100, 2),
            "non_aggressive_accuracy": round(non_agg_correct / max(1, len(non_agg_graded)) * 100, 2),
            "total_correct": correct,
            "total_graded": len(graded),
            "aggressive_correct": agg_correct,
            "aggressive_total": len(agg_graded),
            "non_aggressive_correct": non_agg_correct,
            "non_aggressive_total": len(non_agg_graded),
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
    print(f"Judge: {args.judge} ({args.judge_model})")
    print(f"Graded: {len(graded)}/{len(grades)}")
    print(f"\nOverall accuracy: {output['aggregate']['overall_accuracy']:.1f}% ({correct}/{len(graded)})")
    print(f"  Aggressive clips:     {output['aggregate']['aggressive_accuracy']:.1f}% ({agg_correct}/{len(agg_graded)})")
    print(f"  Non-aggressive clips: {output['aggregate']['non_aggressive_accuracy']:.1f}% ({non_agg_correct}/{len(non_agg_graded)})")
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
