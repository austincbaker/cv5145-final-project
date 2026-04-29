#!/usr/bin/env python3
"""Combine all tables and sample questions into a single markdown report."""

import argparse
import json
import os
import subprocess
import sys


def run_script(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: {' '.join(cmd)} failed:\n{result.stderr}", file=sys.stderr)
        return ""
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Generate a combined markdown report")
    parser.add_argument("-a", "--annotations", default="annotations.json",
                        help="Path to annotations JSON file")
    parser.add_argument("-q", "--questions", default=None,
                        help="Path to generated questions JSON file")
    parser.add_argument("-e", "--evaluation", nargs="*", default=None,
                        help="Path(s) to evaluation JSON file(s)")
    parser.add_argument("-f", "--frames-dir", default=None,
                        help="Path to extracted frames directory (embeds images)")
    parser.add_argument("-n", "--sample-count", type=int, default=2,
                        help="Number of sample questions per category (default: 2)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed for sample questions")
    parser.add_argument("-o", "--output", default="report.md",
                        help="Output markdown file (default: report.md)")
    args = parser.parse_args()

    sections = []

    sections.append("# Bullying Video Understanding Benchmark Report\n")

    # --- Annotation stats ---
    sections.append("# Part 1: Annotation Dataset Statistics\n")
    output = run_script([sys.executable, "annotation_stats.py", args.annotations])
    if output:
        sections.append(output)

    # --- Generated questions stats ---
    if args.questions:
        sections.append("---\n")
        sections.append("# Part 2: Generated Questions Statistics\n")
        output = run_script([sys.executable, "annotation_stats.py", args.annotations,
                             "-q", args.questions])
        for line in output.splitlines():
            if line.startswith("# Annotations Dataset"):
                continue
            if line.startswith("# Generated Questions"):
                sections.append(line + "\n")
                continue
            sections.append(line + "\n")

    # --- Sample questions ---
    if args.questions:
        sections.append("\n---\n")
        sections.append("# Part 3: Sample Questions\n")
        cmd = [sys.executable, "sample_questions.py", args.questions,
               "-n", str(args.sample_count)]
        if args.seed is not None:
            cmd.extend(["-s", str(args.seed)])
        output = run_script(cmd)
        if output:
            if args.frames_dir:
                enhanced_lines = []
                for line in output.splitlines():
                    enhanced_lines.append(line)
                    if line.startswith("### Sample") and "—" in line:
                        video_name = line.split("`")[1] if "`" in line else None
                        if video_name:
                            base = os.path.splitext(video_name)[0]
                            frame_files = sorted([
                                f for f in os.listdir(args.frames_dir)
                                if f.startswith(base + "_frame") and f.endswith(".jpg")
                            ]) if os.path.isdir(args.frames_dir) else []
                            if frame_files:
                                enhanced_lines.append("")
                                for ff in frame_files:
                                    rel_path = os.path.join(args.frames_dir, ff)
                                    enhanced_lines.append(f"![{ff}]({rel_path})")
                                enhanced_lines.append("")
                sections.append("\n".join(enhanced_lines))
            else:
                sections.append(output)

    # --- Evaluation results ---
    if args.evaluation:
        sections.append("\n---\n")
        sections.append("# Part 4: Model Evaluation Results\n")
        if len(args.evaluation) == 1:
            output = run_script([sys.executable, "generate_result_table.py", args.evaluation[0]])
        else:
            combined = []
            for eval_path in args.evaluation:
                with open(eval_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    combined.extend(data)
                else:
                    combined.append(data)
            tmp_path = "/tmp/combined_eval_report.json"
            with open(tmp_path, "w") as f:
                json.dump(combined, f)
            output = run_script([sys.executable, "generate_result_table.py", tmp_path])
        if output:
            sections.append(output)

    report = "\n".join(sections)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
