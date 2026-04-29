#!/usr/bin/env python3
"""Generate a markdown table of statistics from the annotations dataset."""

import json
import sys
from collections import Counter
from pathlib import Path


def load_annotations(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    raise ValueError(f"Unexpected JSON structure in {path}")


def normalize_field(value) -> list[str]:
    """Normalize a field value into a list of individual string entries."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v and str(v).strip()]
    if isinstance(value, str) and value.strip():
        stripped = value.strip()
        if stripped.lower() in ("none", "n/a", ""):
            return []
        return [stripped]
    return []


def count_people(value) -> int:
    return len(normalize_field(value))


def is_normal_video(entry: dict) -> bool:
    fn = entry.get("file_name", entry.get("video_name", ""))
    return "normal" in fn.lower()


def main(path: str = "annotations.json"):
    annotations = load_annotations(path)
    total = len(annotations)
    non_normal = [e for e in annotations if not is_normal_video(e)]
    normal_count = total - len(non_normal)

    # --- Action stats ---
    action_counter = Counter()
    missing_action = 0
    for e in annotations:
        action = e.get("action", "").strip()
        if action and action.lower() not in ("none", "n/a", ""):
            action_counter[action] += 1
    for e in non_normal:
        action = e.get("action", "").strip()
        if not action or action.lower() in ("none", "n/a", ""):
            missing_action += 1

    # --- Environment stats ---
    env_counter = Counter()
    missing_env = 0
    for e in annotations:
        env = e.get("environment", "").strip()
        if env and env.lower() not in ("none", "n/a", ""):
            env_counter[env] += 1
    for e in non_normal:
        env = e.get("environment", "").strip()
        if not env or env.lower() in ("none", "n/a", ""):
            missing_env += 1

    # --- Role stats ---
    role_fields = {"aggressor": "aggressor", "victim": "victim", "bystanders": "bystander"}
    role_stats = {}
    for raw_key, label in role_fields.items():
        present = 0
        missing = 0
        count_dist = Counter()
        desc_counter = Counter()
        is_list_count = 0
        for e in annotations:
            value = e.get(raw_key)
            people = normalize_field(value)
            n = len(people)
            count_dist[n] += 1
            if n > 0:
                present += 1
                for p in people:
                    desc_counter[p] += 1
            if isinstance(value, list):
                is_list_count += 1
        for e in non_normal:
            value = e.get(raw_key)
            people = normalize_field(value)
            if len(people) == 0:
                missing += 1
        role_stats[label] = {
            "present": present,
            "missing": missing,
            "count_dist": count_dist,
            "unique_descriptions": len(desc_counter),
            "top_descriptions": desc_counter.most_common(10),
            "is_list_count": is_list_count,
        }

    # --- People per video ---
    people_per_video = []
    for e in annotations:
        n = 0
        for key in ("aggressor", "victim", "bystanders"):
            n += count_people(e.get(key))
        people_per_video.append(n)
    people_dist = Counter(people_per_video)

    # --- File name prefix (source dataset) ---
    prefix_counter = Counter()
    for e in annotations:
        fn = e.get("file_name", "")
        # Extract prefix before the first underscore group that looks like a source
        parts = fn.rsplit("_", 1)
        if len(parts) >= 2:
            # e.g., "punch_chatgpt_025.mp4" -> "punch"
            action_prefix = fn.split("_")[0]
            prefix_counter[action_prefix] += 1

    # --- Print markdown ---
    print("# Annotations Dataset Statistics\n")
    print(f"**Source file:** `{path}`\n")
    print(f"**Total videos:** {total}  ")
    print(f"**Normal videos (excluded from missing counts):** {normal_count}  ")
    print(f"**Non-normal videos:** {len(non_normal)}\n")

    # Overall summary
    print("## Overview\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Total videos | {total} |")
    print(f"| Normal videos | {normal_count} |")
    print(f"| Non-normal videos | {len(non_normal)} |")
    print(f"| Unique actions | {len(action_counter)} |")
    print(f"| Unique environments | {len(env_counter)} |")
    for label, stats in role_stats.items():
        print(f"| Unique {label} descriptions | {stats['unique_descriptions']} |")
    print(f"| Videos missing action | {missing_action} / {len(non_normal)} ({missing_action/len(non_normal)*100:.1f}%) |")
    print(f"| Videos missing environment | {missing_env} / {len(non_normal)} ({missing_env/len(non_normal)*100:.1f}%) |")
    for label, stats in role_stats.items():
        print(f"| Videos missing {label} | {stats['missing']} / {len(non_normal)} ({stats['missing']/len(non_normal)*100:.1f}%) |")
    print()

    # Action distribution
    print("## Action Distribution\n")
    print("| Action | Count | % of Total |")
    print("|--------|------:|-----------:|")
    for action, count in action_counter.most_common():
        print(f"| {action} | {count} | {count/total*100:.1f}% |")
    print()

    # Environment distribution
    print("## Environment Distribution\n")
    print("| Environment | Count | % of Total |")
    print("|-------------|------:|-----------:|")
    for env, count in env_counter.most_common():
        print(f"| {env} | {count} | {count/total*100:.1f}% |")
    print()

    # Role presence
    print("## Role Presence\n")
    print("| Role | Present | Missing | % Present |")
    print("|------|--------:|--------:|----------:|")
    for label, stats in role_stats.items():
        pct = stats["present"] / total * 100
        print(f"| {label.capitalize()} | {stats['present']} | {stats['missing']} | {pct:.1f}% |")
    print()

    # Count per video for each role
    print("## People Count Per Video (by role)\n")
    for label, stats in role_stats.items():
        print(f"### {label.capitalize()}\n")
        print(f"| Count per video | Videos |")
        print(f"|----------------:|-------:|")
        for n in sorted(stats["count_dist"].keys()):
            print(f"| {n} | {stats['count_dist'][n]} |")
        print()

    # Total people per video
    print("## Total People Per Video\n")
    print("| People count | Videos |")
    print("|-------------:|-------:|")
    for n in sorted(people_dist.keys()):
        print(f"| {n} | {people_dist[n]} |")
    avg_people = sum(people_per_video) / total
    print(f"\n**Average people per video:** {avg_people:.2f}\n")

    # Top person descriptions per role
    print("## Top 10 Descriptions Per Role\n")
    for label, stats in role_stats.items():
        print(f"### {label.capitalize()}\n")
        print("| Description | Count |")
        print("|-------------|------:|")
        for desc, count in stats["top_descriptions"]:
            print(f"| {desc} | {count} |")
        print()

    # Video name prefixes (action types from filenames)
    print("## Videos by Action Prefix (from filename)\n")
    print("| Prefix | Count | % of Total |")
    print("|--------|------:|-----------:|")
    for prefix, count in prefix_counter.most_common():
        print(f"| {prefix} | {count} | {count/total*100:.1f}% |")
    print()


def generated_questions_stats(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    questions_by_video = data.get("questions_by_video", {})

    questions = []
    for video, qs in questions_by_video.items():
        questions.extend(qs)

    total = len(questions)

    secondary_types = {
        "role_count_aggressor", "role_count_victim", "role_count_bystander",
        "compound_aggressor_victim_count", "compound_victim_bystander_count",
        "compound_action_location",
    }

    primary = [q for q in questions if q["question_type"] not in secondary_types]
    secondary = [q for q in questions if q["question_type"] in secondary_types]

    print("# Generated Questions Statistics (Primary Only)\n")
    print(f"**Source file:** `{path}`\n")
    print(f"**Total questions (all):** {total}  ")
    print(f"**Primary questions:** {len(primary)}  ")
    print(f"**Secondary questions:** {len(secondary)}\n")

    if metadata:
        print("## Generation Metadata\n")
        print("| Parameter | Value |")
        print("|-----------|-------|")
        for key in ("num_distractors", "trick_probability", "num_videos",
                     "generation_method", "seed", "sample"):
            if key in metadata:
                print(f"| {key} | {metadata[key]} |")
        dist = metadata.get("distribution_config", {})
        if dist:
            for cat, count in dist.items():
                print(f"| questions_per_video ({cat}) | {count} |")
        print()

    # --- Answer count distribution ---
    answer_counts = Counter(len(q["answers"]) for q in primary)
    print("## Answer Count Distribution\n")
    print("| Number of Answers | Questions | % of Primary |")
    print("|------------------:|----------:|-------------:|")
    count_4 = answer_counts.get(4, 0)
    count_8 = answer_counts.get(8, 0)
    other = len(primary) - count_4 - count_8
    for n in sorted(answer_counts.keys()):
        pct = answer_counts[n] / len(primary) * 100
        print(f"| {n} | {answer_counts[n]} | {pct:.1f}% |")
    print()
    print(f"**4 answers:** {count_4} ({count_4/len(primary)*100:.1f}%)  ")
    print(f"**8 answers:** {count_8} ({count_8/len(primary)*100:.1f}%)  ")
    print(f"**Other:** {other} ({other/len(primary)*100:.1f}%)\n")

    # --- Questions per type ---
    type_counter = Counter(q["question_type"] for q in primary)
    print("## Questions Per Type\n")
    print("| Question Type | Count | % of Primary |")
    print("|---------------|------:|-------------:|")
    for qtype, count in type_counter.most_common():
        pct = count / len(primary) * 100
        print(f"| {qtype} | {count} | {pct:.1f}% |")
    print()

    # --- Answer count distribution per type ---
    print("## Answer Counts by Question Type\n")
    print("| Question Type | Answers | Questions |")
    print("|---------------|--------:|----------:|")
    type_answer_counts = {}
    for q in primary:
        qt = q["question_type"]
        n = len(q["answers"])
        type_answer_counts.setdefault(qt, Counter())[n] += 1
    for qt in sorted(type_answer_counts.keys()):
        for n in sorted(type_answer_counts[qt].keys()):
            print(f"| {qt} | {n} | {type_answer_counts[qt][n]} |")
    print()

    # --- Trick question stats ---
    trick_count = sum(1 for q in primary if q.get("is_trick", False))
    print("## Trick Questions\n")
    print("| Metric | Value |")
    print("|--------|------:|")
    print(f"| Trick questions | {trick_count} |")
    print(f"| Normal questions | {len(primary) - trick_count} |")
    if len(primary) > 0:
        print(f"| Trick rate | {trick_count/len(primary)*100:.1f}% |")
    print()

    trick_by_type = Counter(q["question_type"] for q in primary if q.get("is_trick", False))
    if trick_by_type:
        print("### Trick Questions by Type\n")
        print("| Question Type | Trick | Total | Trick Rate |")
        print("|---------------|------:|------:|-----------:|")
        for qt, total_qt in type_counter.most_common():
            tricks = trick_by_type.get(qt, 0)
            rate = tricks / total_qt * 100 if total_qt > 0 else 0
            print(f"| {qt} | {tricks} | {total_qt} | {rate:.1f}% |")
        print()

    # --- Correct answer position distribution ---
    position_counter = Counter(q["correct_index"] for q in primary)
    print("## Correct Answer Position Distribution\n")
    print("| Position (0-indexed) | Count | % |")
    print("|---------------------:|------:|--:|")
    for pos in sorted(position_counter.keys()):
        count = position_counter[pos]
        pct = count / len(primary) * 100
        print(f"| {pos} | {count} | {pct:.1f}% |")
    print()

    # --- Unique answers per type ---
    print("## Answer Pool Diversity\n")
    print("| Question Type | Unique Answers | Avg Answers/Question | Most Common Answer | Its Count |")
    print("|---------------|---------------:|---------------------:|--------------------:|----------:|")
    for qt, _ in type_counter.most_common():
        qs = [q for q in primary if q["question_type"] == qt]
        all_answers = Counter()
        total_answers = 0
        for q in qs:
            for a in q["answers"]:
                all_answers[a] += 1
                total_answers += 1
        avg = total_answers / len(qs) if qs else 0
        top_answer, top_count = all_answers.most_common(1)[0] if all_answers else ("N/A", 0)
        if len(top_answer) > 50:
            top_answer = top_answer[:47] + "..."
        print(f"| {qt} | {len(all_answers)} | {avg:.1f} | {top_answer} | {top_count} |")
    print()

    # --- Videos with questions ---
    videos_with_qs = len(questions_by_video)
    qs_per_video = [len(qs) for qs in questions_by_video.values()]
    primary_per_video = []
    for video, qs in questions_by_video.items():
        n = sum(1 for q in qs if q["question_type"] not in secondary_types)
        primary_per_video.append(n)
    print("## Coverage\n")
    print("| Metric | Value |")
    print("|--------|------:|")
    print(f"| Videos with questions | {videos_with_qs} |")
    print(f"| Total questions per video (min) | {min(qs_per_video)} |")
    print(f"| Total questions per video (max) | {max(qs_per_video)} |")
    print(f"| Total questions per video (avg) | {sum(qs_per_video)/len(qs_per_video):.1f} |")
    print(f"| Primary questions per video (min) | {min(primary_per_video)} |")
    print(f"| Primary questions per video (max) | {max(primary_per_video)} |")
    print(f"| Primary questions per video (avg) | {sum(primary_per_video)/len(primary_per_video):.1f} |")
    print()

    # --- Duplicate detection ---
    seen = Counter()
    for q in primary:
        key = (q["video_name"], q["question_type"], q["prompt"])
        seen[key] += 1
    duplicates = {k: v for k, v in seen.items() if v > 1}
    print("## Duplicates\n")
    if duplicates:
        print(f"**{len(duplicates)} duplicate question(s) found:**\n")
        print("| Video | Type | Count |")
        print("|-------|------|------:|")
        for (video, qt, _), count in sorted(duplicates.items(), key=lambda x: -x[1])[:20]:
            print(f"| {video} | {qt} | {count} |")
    else:
        print("No duplicate questions found.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate annotation and question statistics")
    parser.add_argument("annotations", nargs="?", default="annotations.json",
                        help="Path to annotations JSON file (default: annotations.json)")
    parser.add_argument("-q", "--questions", default=None,
                        help="Path to generated questions JSON file (adds primary question stats)")
    args = parser.parse_args()

    main(args.annotations)
    if args.questions:
        print("\n---\n")
        generated_questions_stats(args.questions)
