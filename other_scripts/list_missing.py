#!/usr/bin/env python3
"""List videos missing a specific annotation field."""

import argparse
import json
import sys


def load_annotations(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    raise ValueError(f"Unexpected JSON structure in {path}")


NONE_VALUES = {"none", "n/a", ""}

FIELD_ALIASES = {
    "bystanders": "bystanders",
    "bystander": "bystanders",
}


def is_normal_video(entry: dict) -> bool:
    fn = entry.get("file_name", entry.get("video_name", ""))
    return "normal" in fn.lower()


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NONE_VALUES
    if isinstance(value, list):
        return not any(
            isinstance(v, str) and v.strip() and v.strip().lower() not in NONE_VALUES
            for v in value
        )
    return False


def main():
    parser = argparse.ArgumentParser(description="List videos missing an annotation field")
    parser.add_argument("--key", "-k", required=True,
                        help="Annotation field to check (action, aggressor, victim, bystander, environment)")
    parser.add_argument("--annotations", "-a", default="annotations.json",
                        help="Path to annotations JSON file (default: annotations.json)")
    args = parser.parse_args()

    key = FIELD_ALIASES.get(args.key, args.key)
    annotations = load_annotations(args.annotations)

    valid_keys = {k for e in annotations for k in e.keys()} | set(FIELD_ALIASES.keys())
    if key not in valid_keys:
        print(f"Unknown key '{args.key}'. Available keys: {', '.join(sorted(valid_keys))}", file=sys.stderr)
        sys.exit(1)

    non_normal = [e for e in annotations if not is_normal_video(e)]
    missing = [e for e in non_normal if is_missing(e.get(key))]

    print(f"Videos missing '{args.key}': {len(missing)} / {len(non_normal)} (excluding {len(annotations) - len(non_normal)} normal videos)\n")
    for e in missing:
        print(e.get("file_name", e.get("video_name", "unknown")))


if __name__ == "__main__":
    main()
