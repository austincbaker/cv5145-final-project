#!/usr/bin/env python3
"""Combine split question files into a single file for training."""
import json
import sys
from pathlib import Path

def main():
    parts = sorted(Path(".").glob("generated_questions_freq_inv_part*of3.json"))
    if not parts:
        print("No generated_questions_freq_inv_part*of3.json found in cwd")
        sys.exit(1)

    combined = {"metadata": {}, "questions_by_video": {}}
    for p in parts:
        with open(p) as f:
            d = json.load(f)
        combined["questions_by_video"].update(d["questions_by_video"])
        if not combined["metadata"]:
            combined["metadata"] = d.get("metadata", {})

    total_q = sum(len(qs) for qs in combined["questions_by_video"].values())
    print(f"Combined {len(parts)} parts: {len(combined['questions_by_video'])} videos, {total_q} questions")

    out = Path("train_model/data/generated_questions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(combined, f)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
