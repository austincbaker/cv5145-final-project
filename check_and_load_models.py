#!/usr/bin/env python3
"""
Check which VLM models are cached and download missing ones.

This script examines the HuggingFace cache to see which models are already
downloaded, then downloads any missing models from the registry.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import scan_cache_dir, snapshot_download

# Import model shortcuts directly from the model_loader registry module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'prompt_generator', 'evaluation'))
from model_loader.registry import MODEL_SHORTCUTS


def check_model_cached(model_path: str) -> bool:
    """Check if a model is already in the HuggingFace cache."""
    try:
        cache_info = scan_cache_dir()

        # Extract repo name from full path (e.g., "Qwen/Qwen2.5-VL-72B-Instruct" -> "Qwen2.5-VL-72B-Instruct")
        repo_name = model_path.split('/')[-1]

        for repo in cache_info.repos:
            if repo_name.lower() in repo.repo_id.lower() or model_path.lower() == repo.repo_id.lower():
                # Check if it has any revisions (indicating it's been downloaded)
                if len(repo.revisions) > 0:
                    return True

        return False
    except Exception as e:
        print(f"Warning: Could not scan cache for {model_path}: {e}")
        return False


def get_model_size_estimate(model_path: str) -> str:
    """Estimate model size based on naming convention."""
    path_lower = model_path.lower()

    if "90b" in path_lower or "78b" in path_lower or "72b" in path_lower:
        return "~150-180 GB"
    elif "15b" in path_lower or "11b" in path_lower:
        return "~30-40 GB"
    elif "8b" in path_lower or "9b" in path_lower or "7b" in path_lower:
        return "~15-20 GB"
    elif "3b" in path_lower or "2b" in path_lower:
        return "~5-10 GB"
    else:
        return "Unknown"


def download_model(model_path: str, cache_dir: str | None = None) -> bool:
    """Download a model from HuggingFace Hub."""
    try:
        print(f"\n{'='*60}")
        print(f"Downloading: {model_path}")
        print(f"Estimated size: {get_model_size_estimate(model_path)}")
        print(f"{'='*60}\n")

        snapshot_download(
            repo_id=model_path,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        print(f"\n✓ Successfully downloaded: {model_path}\n")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {model_path}: {e}\n")
        return False


def main():
    print("="*60)
    print("VLM Model Cache Checker and Loader")
    print("="*60)
    print()

    # Get cache directory
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        print(f"Using HuggingFace cache: {cache_dir}")
    else:
        print(f"Using default HuggingFace cache: ~/.cache/huggingface")
    print()

    # Get all models from registry
    all_models = list(MODEL_SHORTCUTS.values())

    print(f"Checking {len(all_models)} models from registry...\n")

    cached_models = []
    missing_models = []

    # Check each model
    for shortcut, model_path in MODEL_SHORTCUTS.items():
        is_cached = check_model_cached(model_path)
        status = "✓ CACHED" if is_cached else "✗ MISSING"
        size = get_model_size_estimate(model_path)

        print(f"{status:12} | {shortcut:20} | {model_path:50} | {size}")

        if is_cached:
            cached_models.append((shortcut, model_path))
        else:
            missing_models.append((shortcut, model_path))

    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total models: {len(all_models)}")
    print(f"Cached: {len(cached_models)}")
    print(f"Missing: {len(missing_models)}")
    print()

    if not missing_models:
        print("✓ All models are already cached!")
        return 0

    # Ask to download (or auto-download if --auto flag is present)
    auto_download = "--auto" in sys.argv or "--download" in sys.argv

    if not auto_download:
        print("Missing models:")
        for shortcut, model_path in missing_models:
            size = get_model_size_estimate(model_path)
            print(f"  - {shortcut:20} ({model_path}) - {size}")
        print()
        print("To download missing models, run with --download flag")
        print("Example: python check_and_load_models.py --download")
        return 0

    # Download missing models
    print("Downloading missing models...")
    print(f"Total estimated download: ~{sum_sizes(missing_models)}")
    print()

    successful = []
    failed = []

    for i, (shortcut, model_path) in enumerate(missing_models, 1):
        print(f"\n[{i}/{len(missing_models)}] Processing: {shortcut}")

        if download_model(model_path, cache_dir):
            successful.append((shortcut, model_path))
        else:
            failed.append((shortcut, model_path))

    # Final summary
    print()
    print("="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        print("Successfully downloaded:")
        for shortcut, model_path in successful:
            print(f"  ✓ {shortcut} ({model_path})")
        print()

    if failed:
        print("Failed to download:")
        for shortcut, model_path in failed:
            print(f"  ✗ {shortcut} ({model_path})")
        print()
        return 1

    print("✓ All models are now cached!")
    return 0


def sum_sizes(models: list[tuple[str, str]]) -> str:
    """Estimate total download size."""
    total_gb = 0
    for _, model_path in models:
        size_str = get_model_size_estimate(model_path)
        if "150-180" in size_str:
            total_gb += 165
        elif "30-40" in size_str:
            total_gb += 35
        elif "15-20" in size_str:
            total_gb += 17
        elif "5-10" in size_str:
            total_gb += 7

    return f"{total_gb} GB"


if __name__ == "__main__":
    sys.exit(main())
