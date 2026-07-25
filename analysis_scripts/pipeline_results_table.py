#!/usr/bin/env python3
"""Generate markdown tables summarizing the progressive fine-tuning pipeline results.

Handles two result formats:
  - Combined: each part file has all stages as top-level keys (e.g. {base, sft, cot, adpo})
  - Separate: each part file has a single stage key matching the filename prefix

Usage:
    python analysis_scripts/pipeline_results_table.py
    python analysis_scripts/pipeline_results_table.py --experiments 5pct_adaptive_v1 10pct_adaptive_v1
    python analysis_scripts/pipeline_results_table.py -o analysis_scripts/output/markdown/pipeline_results.md
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


SECONDARY_QUESTION_TYPES = {
    "compound_action_location",
    "role_count_victim",
    "role_count_aggressor",
    "role_count_bystander",
    "compound_aggressor_victim_count",
}


def discover_experiments(base_dir: str) -> list[str]:
    experiments = []
    for d in sorted(Path(base_dir).iterdir()):
        if d.is_dir() and (d / "eval").is_dir():
            experiments.append(d.name)
    return experiments


def load_experiment(eval_dir: Path) -> dict[str, dict[str, dict[str, int]]]:
    """Load eval results, preferring deduplicated merged file over raw parts.

    Returns: {stage: {question_type: {correct, total}}}
    """
    stage_data: dict[str, dict[str, dict[str, int]]] = {}

    merged = eval_dir / "results_merged.json"
    if merged.exists():
        files = [merged]
    else:
        files = sorted(eval_dir.glob("results_part*.json"))
        files = [f for f in files if "_detailed" not in f.name]

    if not files:
        return stage_data

    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        for stage, stage_vals in data.items():
            if not isinstance(stage_vals, dict) or "by_question_type" not in stage_vals:
                continue
            if stage not in stage_data:
                stage_data[stage] = defaultdict(lambda: {"correct": 0, "total": 0})
            for qt, vals in stage_vals["by_question_type"].items():
                stage_data[stage][qt]["correct"] += vals["correct"]
                stage_data[stage][qt]["total"] += vals["total"]

    return {s: dict(v) for s, v in stage_data.items()}


def compute_accuracy(by_qt: dict[str, dict[str, int]], question_types: set[str] | None = None) -> tuple[int, int, float]:
    correct = 0
    total = 0
    for qt, vals in by_qt.items():
        if question_types is not None and qt not in question_types:
            continue
        correct += vals["correct"]
        total += vals["total"]
    acc = correct / max(1, total) * 100
    return correct, total, acc


def format_table(exp_name: str, stage_data: dict, all_qtypes: list[str], base_model: str = "") -> str:
    lines = []
    title_prefix = f"{exp_name} ({base_model})" if base_model else exp_name
    lines.append(f"## {title_prefix}")
    lines.append("")

    stage_order = []
    for s in ["base", "sft", "cot", "dpo", "adpo"]:
        if s in stage_data:
            stage_order.append(s)

    # Overall accuracy table
    lines.append(f"### {title_prefix} -- Overall Accuracy")
    lines.append("")
    lines.append("| Stage | Correct | Total | Accuracy | Delta |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")

    prev_acc = None
    for stage in stage_order:
        correct, total, acc = compute_accuracy(stage_data[stage])
        if prev_acc is not None:
            delta = f"{acc - prev_acc:+.2f}pp"
        else:
            delta = "--"
        lines.append(f"| {stage.upper()} | {correct:,} | {total:,} | {acc:.2f}% | {delta} |")
        prev_acc = acc

    # Primary vs secondary
    primary_qtypes = {qt for qt in all_qtypes if qt not in SECONDARY_QUESTION_TYPES}
    secondary_qtypes = {qt for qt in all_qtypes if qt in SECONDARY_QUESTION_TYPES}

    if primary_qtypes and secondary_qtypes:
        lines.append("")
        lines.append(f"### {title_prefix} -- Primary vs Secondary")
        lines.append("")
        lines.append("| Stage | Primary | Secondary |")
        lines.append("| :--- | ---: | ---: |")
        for stage in stage_order:
            _, _, p_acc = compute_accuracy(stage_data[stage], primary_qtypes)
            _, _, s_acc = compute_accuracy(stage_data[stage], secondary_qtypes)
            lines.append(f"| {stage.upper()} | {p_acc:.2f}% | {s_acc:.2f}% |")

    # Per question type table
    lines.append("")
    lines.append(f"### {title_prefix} -- Per Question Type")
    lines.append("")
    header = "| Question Type |"
    separator = "| :--- |"
    for stage in stage_order:
        header += f" {stage.upper()} |"
        separator += " ---: |"
    lines.append(header)
    lines.append(separator)

    for qt in sorted(all_qtypes):
        row = f"| {qt} |"
        for stage in stage_order:
            vals = stage_data[stage].get(qt)
            if vals and vals["total"] > 0:
                acc = vals["correct"] / vals["total"] * 100
                row += f" {acc:.1f}% ({vals['correct']}/{vals['total']}) |"
            else:
                row += " -- |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default="train_model/experiments")
    parser.add_argument("--experiments", nargs="+", default=None, help="Experiment names to include (default: all)")
    parser.add_argument("--base-model", default="InternVL2.5-8B", help="Base model name for titles")
    parser.add_argument("-o", "--output", default=None, help="Output markdown path")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if args.experiments:
        exp_names = args.experiments
    else:
        exp_names = discover_experiments(base)

    all_sections = []
    all_sections.append("# Progressive Fine-Tuning Pipeline Results")
    all_sections.append("")

    for exp_name in exp_names:
        eval_dir = base / exp_name / "eval"
        if not eval_dir.is_dir():
            print(f"Skipping {exp_name}: no eval/ directory")
            continue

        stage_data = load_experiment(eval_dir)
        if not stage_data:
            print(f"Skipping {exp_name}: no result files")
            continue

        all_qtypes = set()
        for by_qt in stage_data.values():
            all_qtypes.update(by_qt.keys())

        section = format_table(exp_name, stage_data, list(all_qtypes), args.base_model)
        all_sections.append(section)
        print(f"Processed {exp_name}: {len(stage_data)} stages, {len(all_qtypes)} question types")

    output_text = "\n".join(all_sections)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\nWrote: {args.output}")
    else:
        print()
        print(output_text)


if __name__ == "__main__":
    main()
