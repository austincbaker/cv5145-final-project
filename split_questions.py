"""Split generated_questions.json into N parts by video."""
import json
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to generated_questions.json")
parser.add_argument("-n", type=int, default=4, help="Number of parts")
args = parser.parse_args()

with open(args.input, encoding="utf-8") as f:
    data = json.load(f)

metadata = data["metadata"]
videos = list(data["questions_by_video"].items())
total = len(videos)
chunk = math.ceil(total / args.n)

for i in range(args.n):
    start = i * chunk
    end = min((i + 1) * chunk, total)
    part = {
        "metadata": metadata,
        "questions_by_video": dict(videos[start:end]),
    }
    q_count = sum(len(qs) for qs in part["questions_by_video"].values())
    out = args.input.replace(".json", "_part%dof%d.json" % (i + 1, args.n))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(part, f, indent=2, ensure_ascii=False)
    print("%s: %d videos, %d questions" % (out, end - start, q_count))
