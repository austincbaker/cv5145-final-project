#!/usr/bin/env python3
"""
Merge compound_action_aggressor eval results into existing combined result files.

Takes actagg eval JSON(s) (single file or split parts) for a model and appends
the results to the corresponding combined_results/*.json file, updating
metadata counts and per-type accuracy breakdowns.

Usage:
    # Single eval file:
    python analysis_scripts/merge_actagg_results.py \
        --actagg eval_parts/actagg_results/actagg_InternVL2.5-8B_eval.json \
        --combined combined_results/InternVL2.5-8B_combined.json

    # Multiple parts:
    python analysis_scripts/merge_actagg_results.py \
        --actagg eval_parts/actagg_results/actagg_InternVL3_p1_eval.json \
                 eval_parts/actagg_results/actagg_InternVL3_p2_eval.json \
                 eval_parts/actagg_results/actagg_InternVL3_p3_eval.json \
        --combined combined_results/InternVL3-9B_combined.json

    # Dry run (show what would change):
    python analysis_scripts/merge_actagg_results.py \
        --actagg eval_parts/actagg_results/actagg_InternVL2.5-8B_eval.json \
        --combined combined_results/InternVL2.5-8B_combined.json \
        --dry-run
"""
import argparse
import json
from pathlib import Path


REMOVED_VIDEOS = {
    "bodyslam_facebook_001.mp4",
    "indecent__gesture_2_trim_0.mp4",
    "tackle_1_trim_2.mp4",
    "tackle_2_trim_1.mp4",
}


def remove_excluded_videos(results: list[dict], label: str) -> list[dict]:
    before = len(results)
    filtered = [r for r in results if r.get("video_name") not in REMOVED_VIDEOS]
    removed = before - len(filtered)
    if removed:
        print(f"  Removed {removed} questions from excluded videos "
              f"(tackle/bodyslam/indecent gesture) in {label}")
    return filtered


def _parse_selected_index(response: str) -> int | None:
    """Parse model_response like '4.', 'D', '4. punch; ...' into a 0-based index."""
    if not response:
        return None
    response = response.strip()
    # Try number first: "4." or "4" or "4. some text"
    import re
    m = re.match(r"^(\d+)", response)
    if m:
        return int(m.group(1)) - 1
    # Try letter: "D" or "D." or "D) some text"
    m = re.match(r"^([A-Ha-h])", response)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    return None


def _normalize_result(r: dict) -> dict:
    """Normalize a result entry to have is_correct and model_selected_index."""
    if r.get("is_correct") is not None:
        return r
    if "model_response" not in r:
        return r
    idx = _parse_selected_index(r["model_response"])
    correct_idx = r.get("correct_index")
    r["model_selected_index"] = idx
    r["is_correct"] = (idx == correct_idx) if idx is not None else False
    return r


def load_and_combine_parts(paths: list[str]) -> list[dict]:
    all_results = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        results = d["results"]
        normalized = 0
        for r in results:
            if r.get("is_correct") is None and "model_response" in r:
                normalized += 1
            _normalize_result(r)
        if normalized:
            print(f"  Parsed {normalized} model_response entries in {Path(p).name}")
        all_results.extend(results)
    return all_results


def merge(combined_path: str, actagg_results: list[dict], dry_run: bool = False):
    with open(combined_path, encoding="utf-8") as f:
        combined = json.load(f)

    combined["results"] = remove_excluded_videos(combined["results"], "existing results")
    actagg_results = remove_excluded_videos(actagg_results, "actagg results")

    # Remove any existing compound_action_aggressor to prevent duplication
    existing_actagg = sum(
        1 for r in combined["results"]
        if r["question_type"] == "compound_action_aggressor"
    )
    if existing_actagg:
        print(f"  Replacing {existing_actagg} existing compound_action_aggressor results")
        combined["results"] = [
            r for r in combined["results"]
            if r["question_type"] != "compound_action_aggressor"
        ]

    combined["results"].extend(actagg_results)

    # Recalculate all metadata from the filtered results
    from collections import defaultdict
    primary_types = {
        "primary_action", "role_identification", "aggressor_identification",
        "victim_recognition", "compound_action_aggressor", "compound_action_victims",
        "compound_aggressor_victim", "compound_aggressor_action_victim",
        "sequence_verification",
    }
    secondary_types = {
        "compound_action_location",
        "role_count_victim", "role_count_aggressor", "role_count_bystander",
        "compound_aggressor_victim_count", "compound_victim_bystander_count",
    }

    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in combined["results"]:
        qt = r["question_type"]
        by_type[qt]["total"] += 1
        if r.get("is_correct"):
            by_type[qt]["correct"] += 1

    for qt in by_type:
        t = by_type[qt]["total"]
        c = by_type[qt]["correct"]
        by_type[qt]["accuracy"] = c / t if t else 0

    primary_total = sum(by_type[qt]["total"] for qt in by_type if qt in primary_types)
    primary_correct = sum(by_type[qt]["correct"] for qt in by_type if qt in primary_types)
    secondary_total = sum(by_type[qt]["total"] for qt in by_type if qt in secondary_types)
    secondary_correct = sum(by_type[qt]["correct"] for qt in by_type if qt in secondary_types)

    combined["primary_total_questions"] = primary_total
    combined["primary_correct_count"] = primary_correct
    combined["primary_accuracy"] = primary_correct / primary_total if primary_total else 0
    combined["primary_accuracy_by_type"] = {qt: dict(by_type[qt]) for qt in by_type if qt in primary_types}
    combined["secondary_total_questions"] = secondary_total
    combined["secondary_correct_count"] = secondary_correct
    combined["secondary_accuracy"] = secondary_correct / secondary_total if secondary_total else 0
    combined["total_questions"] = len(combined["results"])

    actagg_stats = by_type.get("compound_action_aggressor", {"total": 0, "correct": 0, "accuracy": 0})
    overall_total = primary_total + secondary_total
    overall_correct = primary_correct + secondary_correct

    print(f"  Action+Aggressor: {actagg_stats['correct']}/{actagg_stats['total']} ({actagg_stats['accuracy']*100:.2f}%)")
    print(f"  Primary: {primary_correct}/{primary_total} ({combined['primary_accuracy']*100:.2f}%)")
    print(f"  Overall: {overall_correct}/{overall_total} ({overall_correct/overall_total*100:.2f}%)")

    if not dry_run:
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {combined_path}")
    else:
        print("  (dry-run, not saved)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actagg", nargs="+", required=True,
                        help="Action+Aggressor eval result JSON(s)")
    parser.add_argument("--combined", required=True,
                        help="Combined results file to merge into")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args()

    print(f"Loading {len(args.actagg)} actagg file(s)...")
    actagg_results = load_and_combine_parts(args.actagg)
    print(f"  {len(actagg_results)} results loaded")

    print(f"Merging into {args.combined}...")
    merge(args.combined, actagg_results, args.dry_run)


if __name__ == "__main__":
    main()
