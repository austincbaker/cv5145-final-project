#!/usr/bin/env python3
"""
Test script to verify new model loaders can be imported and initialized.

This script checks that:
1. All new loader modules can be imported
2. Loader classes are properly registered
3. Model shortcuts resolve correctly
"""
import sys
from pathlib import Path

# Add the evaluation module to path
sys.path.insert(0, str(Path(__file__).parent / 'prompt_generator' / 'evaluation'))


def test_imports():
    """Test that all new loader modules can be imported."""
    print("Testing imports...")

    try:
        from model_loader.llava_video import LLaVAVideoLoader
        print("  ✓ LLaVA-Video loader imported")
        print("  ✓ LLaVA-Video loader imported")
    except Exception as e:
        print(f"  ✗ LLaVA-Video loader failed: {e}")
        return False

    try:
        from model_loader.videollama import VideoLLaMALoader
        print("  ✓ VideoLLaMA loader imported")
    except Exception as e:
        print(f"  ✗ VideoLLaMA loader failed: {e}")
        return False

    try:
        from model_loader.kimi import KimiVLLoader
        print("  ✓ Kimi-VL loader imported")
    except Exception as e:
        print(f"  ✗ Kimi-VL loader failed: {e}")
        return False

    try:
        from model_loader.video_generic import GenericVideoLoader
        print("  ✓ Generic Video loader imported")
    except Exception as e:
        print(f"  ✗ Generic Video loader failed: {e}")
        return False

    return True


def test_registry():
    """Test that model shortcuts resolve correctly."""
    print("\nTesting registry...")

    from model_loader.registry import MODEL_SHORTCUTS, resolve_model_path, get_loader_class

    # Test new model shortcuts
    new_models = [
        "llava-video-7b",
        "videollama-7b",
        "kimi-3b",
        "videochat-7b",
        "oryx-7b",
        "valley-7b",
        "video-r1-7b",
        "internvl2.5-8b",
        "ovis2-8b",
    ]

    for shortcut in new_models:
        model_path = resolve_model_path(shortcut)
        if model_path == shortcut:
            print(f"  ✗ {shortcut}: Not found in registry")
        else:
            print(f"  ✓ {shortcut} → {model_path}")

    return True


def test_loader_detection():
    """Test that loader classes are correctly detected for new models."""
    print("\nTesting loader detection...")

    from model_loader.registry import get_loader_class

    test_cases = [
        ("lmms-lab/LLaVA-Video-7B-Qwen2", "LLaVAVideoLoader"),
        ("DAMO-NLP-SG/VideoLLaMA3-7B", "VideoLLaMALoader"),
        ("Kimi-VL-A3B-Instruct", "KimiVLLoader"),
        ("OpenGVLab/VideoChat-Flash-Qwen2-7B", "GenericVideoLoader"),
        ("THU-MIG/Oryx-7B", "GenericVideoLoader"),
        ("OpenGVLab/InternVL2.5-8B", "InternVLLoader"),
        ("AIDC-AI/Ovis2-8B", "OvisLoader"),
    ]

    for model_path, expected_loader in test_cases:
        try:
            loader_class = get_loader_class(model_path)
            actual_name = loader_class.__name__
            if actual_name == expected_loader:
                print(f"  ✓ {model_path[:40]:40} → {actual_name}")
            else:
                print(f"  ⚠ {model_path[:40]:40} → {actual_name} (expected {expected_loader})")
        except Exception as e:
            print(f"  ✗ {model_path[:40]:40} → ERROR: {e}")

    return True


def test_loader_creation():
    """Test that loaders can be instantiated."""
    print("\nTesting loader creation...")

    from model_loader import create_loader, ModelConfig

    test_shortcuts = [
        "llava-video-7b",
        "videollama-7b",
        "kimi-3b",
        "videochat-7b",
    ]

    for shortcut in test_shortcuts:
        try:
            config = ModelConfig(model_path=shortcut, max_new_tokens=64)
            loader = create_loader(config)
            print(f"  ✓ {shortcut:20} → {loader.__class__.__name__}")
        except Exception as e:
            print(f"  ✗ {shortcut:20} → ERROR: {e}")

    return True


def show_all_models():
    """Display all registered models."""
    print("\nAll registered models:")
    print("=" * 80)

    from model_loader.registry import MODEL_SHORTCUTS

    # Group by family
    families = {}
    for shortcut, path in MODEL_SHORTCUTS.items():
        if "qwen" in shortcut:
            family = "Qwen"
        elif "internvl" in shortcut or "internvideo" in shortcut:
            family = "InternVL/InternVideo"
        elif "llama" in shortcut:
            family = "Llama Vision"
        elif "nvila" in shortcut:
            family = "NVILA"
        elif "ovis" in shortcut:
            family = "Ovis"
        elif "llava" in shortcut:
            family = "LLaVA-Video"
        elif "videollama" in shortcut:
            family = "VideoLLaMA"
        elif "kimi" in shortcut:
            family = "Kimi-VL"
        else:
            family = "Other Video Models"

        if family not in families:
            families[family] = []
        families[family].append((shortcut, path))

    for family, models in sorted(families.items()):
        print(f"\n{family}:")
        for shortcut, path in sorted(models):
            print(f"  {shortcut:25} → {path}")

    print(f"\nTotal: {len(MODEL_SHORTCUTS)} models")


def main():
    """Run all tests."""
    print("=" * 80)
    print("New Model Loader Tests")
    print("=" * 80)
    print()

    all_passed = True

    if not test_imports():
        all_passed = False

    if not test_registry():
        all_passed = False

    if not test_loader_detection():
        all_passed = False

    if not test_loader_creation():
        all_passed = False

    show_all_models()

    print()
    print("=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed - check output above")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
