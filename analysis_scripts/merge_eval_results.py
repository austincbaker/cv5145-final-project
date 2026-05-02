#!/usr/bin/env python3
"""
Merge eval results into an existing combined result file and output a CSV row.

Takes one or more eval result JSONs (any question type) for a model and merges
them into the corresponding combined_results/*.json file. Deduplicates by
(video_name, prompt, question_type). Recalculates all metadata and optionally
prints a CSV row matching baseline_accuracy_summary.csv format.

Usage:
    # Merge missing questions backfill:
    python analysis_scripts/merge_eval_results.py \
        --eval eval_parts/missing_questions/results_missing_qwen2_5_72B.json \
        --combined combined_results/qwen2_5_72B_combined.json

    # Multiple eval files:
    python analysis_scripts/merge_eval_results.py \
        --eval results_part1.json results_part2.json \
        --combined combined_results/model_combined.json

    # Dry run (show what would change):
    python analysis_scripts/merge_eval_results.py \
        --eval results.json \
        --combined combined_results/model_combined.json \
        --dry-run

    # Output CSV row:
    python analysis_scripts/merge_eval_results.py \
        --eval results.json \
        --combined combined_results/model_combined.json \
        --csv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


REMOVED_VIDEOS = {
    "bodyslam_facebook_001.mp4",
    "indecent__gesture_2_trim_0.mp4",
    "tackle_1_trim_2.mp4",
    "tackle_2_trim_1.mp4",
}

PRIMARY_TYPES = {
    "primary_action", "role_identification", "aggressor_identification",
    "victim_recognition", "compound_action_aggressor", "compound_action_victims",
    "compound_aggressor_victim", "compound_aggressor_action_victim",
    "sequence_verification",
}
SECONDARY_TYPES = {
    "compound_action_location",
    "role_count_victim", "role_count_aggressor", "role_count_bystander",
    "compound_aggressor_victim_count", "compound_victim_bystander_count",
}


def _parse_selected_index(response):
    """Parse model_response like '4.', 'D', '4. punch; ...' into a 0-based index."""
    import re
    if not response:
        return None
    response = response.strip()
    m = re.match(r"^(\d+)", response)
    if m:
        return int(m.group(1)) - 1
    m = re.match(r"^([A-Ha-h])", response)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    return None


def _normalize_result(r):
    """Ensure result has is_correct and model_selected_index fields."""
    if r.get("is_correct") is not None and r.get("model_selected_index") is not None:
        return r
    if "model_response" not in r:
        return r
    idx = _parse_selected_index(r["model_response"])
    correct_idx = r.get("correct_index")
    r["model_selected_index"] = idx
    r["is_correct"] = (idx == correct_idx) if idx is not None else False
    if "error" not in r:
        r["error"] = None
    return r


def load_and_combine(paths):
    all_results = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        results = d.get("results", [])
        normalized = 0
        for r in results:
            if r.get("is_correct") is None:
                normalized += 1
            _normalize_result(r)
        if normalized:
            print(f"  Parsed {normalized} raw model_response entries in {Path(p).name}")
        all_results.extend(results)
    return all_results


def merge(combined_path, new_results, dry_run=False, print_csv=False):
    with open(combined_path, encoding="utf-8") as f:
        combined = json.load(f)

    # Remove excluded videos from both
    combined["results"] = [
        r for r in combined["results"]
        if r.get("video_name") not in REMOVED_VIDEOS
    ]
    new_results = [
        r for r in new_results
        if r.get("video_name") not in REMOVED_VIDEOS
    ]

    # Dedup: only add questions not already present
    existing_keys = {
        (r["video_name"], r["prompt"], r["question_type"])
        for r in combined["results"]
    }

    added = 0
    for r in new_results:
        key = (r["video_name"], r["prompt"], r["question_type"])
        if key not in existing_keys:
            combined["results"].append(r)
            existing_keys.add(key)
            added += 1

    print(f"  {len(new_results)} new results, {added} added, "
          f"{len(new_results) - added} duplicates skipped")

    # Recalculate all metadata
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in combined["results"]:
        qt = r["question_type"]
        by_type[qt]["total"] += 1
        if r.get("is_correct"):
            by_type[qt]["correct"] += 1
    for qt in by_type:
        t, c = by_type[qt]["total"], by_type[qt]["correct"]
        by_type[qt]["accuracy"] = c / t if t else 0

    pc = sum(by_type[qt]["correct"] for qt in by_type if qt in PRIMARY_TYPES)
    pt = sum(by_type[qt]["total"] for qt in by_type if qt in PRIMARY_TYPES)
    sc = sum(by_type[qt]["correct"] for qt in by_type if qt in SECONDARY_TYPES)
    st = sum(by_type[qt]["total"] for qt in by_type if qt in SECONDARY_TYPES)

    combined["total_questions"] = len(combined["results"])
    combined["primary_total_questions"] = pt
    combined["primary_correct_count"] = pc
    combined["primary_accuracy"] = pc / pt if pt else 0
    combined["primary_accuracy_by_type"] = {
        qt: dict(by_type[qt]) for qt in by_type if qt in PRIMARY_TYPES
    }
    combined["secondary_total_questions"] = st
    combined["secondary_correct_count"] = sc
    combined["secondary_accuracy"] = sc / st if st else 0
    combined["secondary_accuracy_by_type"] = {
        qt: dict(by_type[qt]) for qt in by_type if qt in SECONDARY_TYPES
    }

    overall_total = pt + st
    overall_correct = pc + sc

    print(f"  Total: {combined['total_questions']}  "
          f"Primary: {pc}/{pt} ({pc/pt*100:.2f}%)  "
          f"Secondary: {sc}/{st} ({sc/st*100:.2f}%)  "
          f"Overall: {overall_correct}/{overall_total} "
          f"({overall_correct/overall_total*100:.2f}%)")

    if print_csv:
        qt_cols = [
            "primary_action", "role_identification", "aggressor_identification",
            "victim_recognition", "compound_action_aggressor",
            "compound_action_victims", "compound_aggressor_victim",
            "compound_aggressor_action_victim", "sequence_verification",
        ]
        single = ["primary_action", "role_identification",
                   "aggressor_identification", "victim_recognition"]
        compound_gh = ["compound_action_aggressor", "compound_action_victims",
                       "compound_aggressor_victim"]
        fine_ij = ["compound_aggressor_action_victim", "sequence_verification"]
        sec_cols = ["compound_action_location", "role_count_victim",
                    "role_count_aggressor", "role_count_bystander",
                    "compound_aggressor_victim_count"]

        def pct(qt):
            if qt not in by_type or by_type[qt]["total"] == 0:
                return ""
            return f"{by_type[qt]['correct']/by_type[qt]['total']*100:.2f}%"

        def sa(types):
            c = sum(by_type.get(qt, {}).get("correct", 0) for qt in types)
            t = sum(by_type.get(qt, {}).get("total", 0) for qt in types)
            return f"{c/t*100:.2f}%" if t else ""

        model_name = Path(combined_path).stem.replace("_combined", "")
        row = [model_name]
        row += [pct(qt) for qt in qt_cols]
        row += [sa(single), sa(compound_gh), sa(fine_ij), sa(list(PRIMARY_TYPES))]
        row += [pct(qt) for qt in sec_cols]
        row += [sa(list(SECONDARY_TYPES)),
                sa(list(PRIMARY_TYPES) + list(SECONDARY_TYPES))]
        print(f"\nCSV row:\n{','.join(row)}")

    if not dry_run:
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {combined_path}")
    else:
        print("  (dry-run, not saved)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval", nargs="+", required=True,
                        help="Eval result JSON(s) to merge in")
    parser.add_argument("--combined", required=True,
                        help="Combined results file to merge into")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--csv", action="store_true",
                        help="Print a CSV row matching baseline_accuracy_summary.csv")
    args = parser.parse_args()

    print(f"Loading {len(args.eval)} eval file(s)...")
    new_results = load_and_combine(args.eval)
    print(f"  {len(new_results)} results loaded")

    print(f"Merging into {args.combined}...")
    merge(args.combined, new_results, args.dry_run, args.csv)


if __name__ == "__main__":
    main()
