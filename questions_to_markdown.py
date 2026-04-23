#!/usr/bin/env python3
"""Convert generated questions JSON to a readable markdown file.

Usage:
    python questions_to_markdown.py generated_questions.json
    python questions_to_markdown.py generated_questions.json -o output.md
"""
import argparse
import json
from pathlib import Path


def convert(questions_path: str, output_path: str, include_secondary: bool = False) -> None:
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    questions_by_video = data.get("questions_by_video", {})

    lines: list[str] = []

    lines.append("# Generated Questions\n")
    lines.append(f"- **Videos:** {metadata.get('num_videos', len(questions_by_video))}")
    lines.append(f"- **Total questions:** {metadata.get('num_questions', '?')}")
    lines.append(f"- **Trick probability:** {metadata.get('trick_probability', '?')}")
    if metadata.get("sample") is not None:
        lines.append(f"- **Sample:** {metadata['sample']} (seed={metadata.get('seed', '?')})")
    lines.append("")
    lines.append("---\n")

    for video_name in sorted(questions_by_video.keys()):
        questions = questions_by_video[video_name]
        lines.append(f"## {video_name}\n")

        filtered = [q for q in questions if include_secondary or not q.get("is_secondary")]
        if not filtered:
            continue

        for i, q in enumerate(filtered, 1):
            trick_tag = " `[TRICK]`" if q.get("is_trick") else ""
            secondary_tag = " `[SECONDARY]`" if q.get("is_secondary") else ""
            # An inverted question is one built by the standalone
            # frequency-inverted builder. Its option_hardness will contain
            # at least one "frequency_saturation" label (the saturation
            # distractors); balanced / hard / easy questions never emit
            # that category.
            hardness = q.get("option_hardness") or []
            inverted_tag = " `[INVERTED]`" if "frequency_saturation" in hardness else ""
            lines.append(
                f"### Q{i}: {q['question_type']}{trick_tag}{secondary_tag}{inverted_tag}\n"
            )
            lines.append(f"**{q['prompt']}**\n")

            for j, answer in enumerate(q["answers"]):
                check = " -- CORRECT" if j == q["correct_index"] else ""
                # Inline the per-option hardness label if present so the
                # reader can see which slot is role_reversal vs saturation.
                h_label = f"  _[{hardness[j]}]_" if j < len(hardness) else ""
                lines.append(f"{j + 1}. {answer}{h_label}{check}")

            lines.append(f"\n*Correct: option {q['correct_index'] + 1} -- {q['correct_answer']}*\n")

        lines.append("---\n")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(questions_by_video)} videos to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert questions JSON to readable markdown")
    parser.add_argument("questions_json", help="Path to generated questions JSON file")
    parser.add_argument("-o", "--output", default=None, help="Output markdown path (default: same name with .md)")
    parser.add_argument("--include-secondary", action="store_true", help="Include secondary questions (excluded by default)")
    args = parser.parse_args()

    output = args.output or str(Path(args.questions_json).with_suffix(".md"))
    convert(args.questions_json, output, include_secondary=args.include_secondary)


if __name__ == "__main__":
    main()
