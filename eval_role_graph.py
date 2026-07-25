"""General-purpose role_graph evaluation using model_loader registry."""
import json
import re
import argparse
import time
from pathlib import Path

from prompt_generator.evaluation.model_loader import create_loader, ModelConfig
from prompt_generator.evaluation.model_loader.registry import get_loader_class

ROLE_GRAPH_PROMPT = r"""
Your primary task is to act as a causal reasoning engine for understanding aggressive scenes in videos. You will observe the persons present and use a causal graph to identify each person's role and the nature of the aggressive interaction.

### Component Definitions

* P (Persons): All individuals visible in the video.
* I (Interaction): Whether a given person is engaged in a direct physical interaction with another person. YES or NO.
* D (Direction): For persons where I = YES: INITIATING or RECEIVING.
* R (Role): Aggressor (I=YES, D=INITIATING), Victim (I=YES, D=RECEIVING), or Bystander (I=NO).
* A (Aggressive Action): The specific type of aggressive behavior, inferred from the Aggressor's motions and Victim's reactions.

### The Causal Graph

P -> I
I (YES) -> D
I (NO) -> R (Bystander)
D (INITIATING) -> R (Aggressor)
D (RECEIVING) -> R (Victim)
R (Aggressor) + R (Victim) -> A

### Your Task

Apply the causal graph to reason about every person you see in the video. Reason through P, I, D, R, A internally. Do not output your reasoning.

From the list of options below, select the answer that most accurately reflects the question asked about the video.
There might also be a choice if none of the other options apply.
Review the video and options carefully. Only one choice is correct.

Output format: reply with the choice number only.
"""

AVOID_VIDEOS = ["aggressive__talking_1_trim_0.mp4"]


def format_prompt(question_text, answers):
    lines = [ROLE_GRAPH_PROMPT.strip(), "", "---", "", "Question: " + question_text, "", "Options:"]
    for i, a in enumerate(answers):
        lines.append("%d. %s" % (i + 1, a))
    lines.extend(["", "Answer with ONLY the option number (e.g., '1' or '2')."])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--frames-dir", default="train_model/data/frames")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    with open(args.questions_json, encoding="utf-8") as f:
        data = json.load(f)
    questions_by_video = data["questions_by_video"]
    print("Loaded questions for %d videos from %s" % (len(questions_by_video), args.questions_json))

    if args.output is None:
        model_short = args.model.split("/")[-1]
        qfile_stem = Path(args.questions_json).stem
        args.output = "%s_%s_role_graph.json" % (model_short, qfile_stem)
    print("Output: %s" % args.output)

    config = ModelConfig(
        model_path=args.model,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    loader = create_loader(config)
    print("Loading model: %s" % args.model)
    loader.load()
    print("Model loaded.")

    from PIL import Image

    def load_frames(video_name, frames_dir, num_frames=8):
        stem = Path(video_name).stem
        frame_dir = Path(frames_dir) / stem
        if not frame_dir.exists():
            return []
        frame_files = sorted(frame_dir.glob("frame_*.jpg"))[:num_frames]
        return [Image.open(f).convert("RGB") for f in frame_files]

    results = []
    checkpoint_path = args.output + ".checkpoint.json"
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            results = json.load(f)
        done_keys = set((r["video_name"], r["prompt"]) for r in results)
        print("Resuming from checkpoint: %d results" % len(results))
    else:
        done_keys = set()

    total_videos = len(questions_by_video)
    for vi, (video_name, questions) in enumerate(questions_by_video.items()):
        if video_name in AVOID_VIDEOS:
            continue

        frames = load_frames(video_name, args.frames_dir, args.num_frames)
        if not frames:
            print("SKIP %s (no frames)" % video_name)
            continue

        print("Processing: %s (%d questions) [%d/%d]" % (video_name, len(questions), vi + 1, total_videos))

        for q in questions:
            key = (q["video_name"], q["prompt"])
            if key in done_keys:
                continue

            prompt = format_prompt(q["prompt"], q["answers"])
            t0 = time.time()
            try:
                response = loader.generate_response(
                    images=frames,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                print("  ERROR: %s" % str(e).encode("ascii", "replace").decode())
                response = ""

            elapsed = time.time() - t0
            response = response.strip()

            results.append({
                "video_name": q["video_name"],
                "question_type": q["question_type"],
                "prompt": q["prompt"],
                "answers": q["answers"],
                "correct_answer": q["correct_answer"],
                "correct_index": q["correct_index"],
                "model_response": response,
                "template": "role_graph",
            })
            done_keys.add(key)

        if (vi + 1) % args.checkpoint_every == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print("  [Checkpoint] %d results" % len(results))

    output = {"total_questions": len(results), "template": "role_graph", "model": args.model, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    correct = sum(1 for r in results if re.search(r"(\d+)", r.get("model_response", "")) and int(re.search(r"(\d+)", r["model_response"]).group(1)) - 1 == r["correct_index"])
    print("Done! %d questions, %d correct (%.2f%%)" % (len(results), correct, 100.0 * correct / max(len(results), 1)))


if __name__ == "__main__":
    main()
