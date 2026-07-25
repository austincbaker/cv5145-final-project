"""Split annotations.json into N parts."""
import json
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to annotations.json")
parser.add_argument("-n", type=int, default=4, help="Number of parts")
args = parser.parse_args()

with open(args.input, encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
chunk = math.ceil(total / args.n)

for i in range(args.n):
    start = i * chunk
    end = min((i + 1) * chunk, total)
    part = data[start:end]
    out = args.input.replace(".json", "_part%dof%d.json" % (i + 1, args.n))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(part, f, indent=2, ensure_ascii=False)
    print("%s: %d annotations" % (out, len(part)))
