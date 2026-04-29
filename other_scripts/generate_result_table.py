import json
import argparse
import sys
import os
from collections import Counter


def format_task_name(task_string):
    """Converts snake_case keys into Title Case strings."""
    return task_string.replace('_', ' ').title()



SECONDARY_TYPES = {
    "role_count_aggressor", "role_count_victim", "role_count_bystander",
    "compound_aggressor_victim_count", "compound_victim_bystander_count",
    "compound_action_location",
}


def backfill_summary(model: dict) -> dict:
    """Compute summary fields from the results array when they are missing."""
    results = model.get('results', [])
    if not results:
        return model

    has_summary = (
        model.get('primary_total_questions')
        and model.get('primary_accuracy') is not None
        and model.get('primary_accuracy_by_type')
    )
    if has_summary:
        return model

    primary_by_type = {}
    secondary_by_type = {}
    videos = set()

    for r in results:
        qt = r.get('question_type', 'unknown')
        is_correct = r.get('is_correct', False)
        videos.add(r.get('video_name'))

        bucket = secondary_by_type if qt in SECONDARY_TYPES else primary_by_type
        if qt not in bucket:
            bucket[qt] = {'total': 0, 'correct': 0}
        bucket[qt]['total'] += 1
        if is_correct:
            bucket[qt]['correct'] += 1

    for bucket in (primary_by_type, secondary_by_type):
        for td in bucket.values():
            td['accuracy'] = td['correct'] / td['total'] if td['total'] > 0 else 0.0

    p_total = sum(td['total'] for td in primary_by_type.values())
    p_correct = sum(td['correct'] for td in primary_by_type.values())
    s_total = sum(td['total'] for td in secondary_by_type.values())
    s_correct = sum(td['correct'] for td in secondary_by_type.values())

    model['primary_total_questions'] = p_total
    model['primary_correct_count'] = p_correct
    model['primary_accuracy'] = p_correct / p_total if p_total > 0 else 0.0
    model['primary_accuracy_by_type'] = primary_by_type
    model['secondary_total_questions'] = s_total
    model['secondary_correct_count'] = s_correct
    model['secondary_accuracy'] = s_correct / s_total if s_total > 0 else 0.0
    model['secondary_accuracy_by_type'] = secondary_by_type
    model['total_videos_evaluated'] = len(videos)

    return model


def generate_markdown_tables(data):
    if isinstance(data, dict):
        models = [data]
    elif isinstance(data, list):
        models = data
    else:
        print("Error: JSON data must be a dictionary or a list of dictionaries.")
        sys.exit(1)

    models = [backfill_summary(m) for m in models]
    models = sorted(models, key=lambda x: x.get('primary_accuracy', 0) or 0)
    model_names = [model.get('model_path', 'Unknown Model') for model in models]
    header_align = lambda n: " | ".join([":---"] * n)

    # --- TABLE 1: Primary Model Overview ---
    headers = ["Metric"] + model_names
    print("### Primary Model Overview\n")
    print(f"| {' | '.join(headers)} |")
    print(f"| {header_align(len(headers))} |")

    def safe_int(m, key):
        v = m.get(key)
        return v if v is not None else 0

    def safe_float(m, key):
        v = m.get(key)
        return v if v is not None else 0.0

    rows = [
        ("Frames Used", [str(m.get('num_frames', 'N/A')) for m in models]),
        ("Total Primary Questions", [f"{safe_int(m, 'primary_total_questions'):,}" for m in models]),
        ("Primary Correct", [f"{safe_int(m, 'primary_correct_count'):,}" for m in models]),
        ("Primary Accuracy", [f"{safe_float(m, 'primary_accuracy') * 100:.2f}%" for m in models]),
        ("Total Secondary Questions", [f"{safe_int(m, 'secondary_total_questions'):,}" for m in models]),
        ("Secondary Correct", [f"{safe_int(m, 'secondary_correct_count'):,}" for m in models]),
        ("Secondary Accuracy", [f"{safe_float(m, 'secondary_accuracy') * 100:.2f}%" for m in models]),
        ("Videos Evaluated", [f"{safe_int(m, 'total_videos_evaluated'):,}" for m in models]),
    ]
    for label, vals in rows:
        print(f"| {label} | {' | '.join(vals)} |")
    print("\n---\n")

    # --- TABLE 2: Primary Task Accuracy Breakdown ---
    all_primary_tasks = set()
    for model in models:
        all_primary_tasks.update((model.get('primary_accuracy_by_type') or {}).keys())
    sorted_primary_tasks = sorted(all_primary_tasks)

    headers = ["Task Type"] + model_names
    print("### Primary Task Accuracy Breakdown\n")
    print(f"| {' | '.join(headers)} |")
    print(f"| {header_align(len(headers))} |")

    for task in sorted_primary_tasks:
        row = [format_task_name(task)]
        for model in models:
            td = (model.get('primary_accuracy_by_type') or {}).get(task)
            if td and 'accuracy' in td:
                row.append(f"{td['accuracy'] * 100:.2f}% ({td['correct']}/{td['total']})")
            else:
                row.append("N/A")
        print(f"| {' | '.join(row)} |")
    print("\n---\n")

    # --- TABLE 3: Secondary Task Accuracy Breakdown ---
    all_secondary_tasks = set()
    for model in models:
        all_secondary_tasks.update((model.get('secondary_accuracy_by_type') or {}).keys())

    if all_secondary_tasks:
        sorted_secondary_tasks = sorted(all_secondary_tasks)
        headers = ["Task Type"] + model_names
        print("### Secondary Task Accuracy Breakdown\n")
        print(f"| {' | '.join(headers)} |")
        print(f"| {header_align(len(headers))} |")

        for task in sorted_secondary_tasks:
            row = [format_task_name(task)]
            for model in models:
                td = (model.get('secondary_accuracy_by_type') or {}).get(task)
                if td and 'accuracy' in td:
                    row.append(f"{td['accuracy'] * 100:.2f}% ({td['correct']}/{td['total']})")
                else:
                    row.append("N/A")
            print(f"| {' | '.join(row)} |")
        print("\n---\n")

    # --- TABLE 4: Error Analysis ---
    for model in models:
        results = model.get('results', [])
        if not results:
            continue

        model_name = model.get('model_path', 'Unknown')
        error_count = sum(1 for r in results if r.get('error'))
        no_response = sum(1 for r in results if r.get('model_selected_index') is None and not r.get('error'))

        if error_count > 0 or no_response > 0:
            print(f"### Error Summary: {model_name}\n")
            print("| Metric | Value |")
            print("| :--- | ---: |")
            print(f"| Total results | {len(results):,} |")
            print(f"| Errors | {error_count:,} |")
            print(f"| No valid response | {no_response:,} |")
            print(f"| Valid responses | {len(results) - error_count - no_response:,} |")
            print("\n---\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown tables from JSON evaluation data.")
    parser.add_argument("filename", help="Path to the JSON file containing the data.")
    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print(f"Error: File '{args.filename}' not found.")
        sys.exit(1)

    try:
        with open(args.filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in file '{args.filename}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    generate_markdown_tables(data)


if __name__ == "__main__":
    main()
