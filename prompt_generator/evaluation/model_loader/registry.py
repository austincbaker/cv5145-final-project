"""
Model registry for vision-language model loaders.

Maps model path substrings to their corresponding loader classes
and provides a shortcut system for CLI convenience.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseVLMLoader

# Shortcut names -> full HuggingFace model paths
MODEL_SHORTCUTS: dict[str, str] = {
    # Qwen variants
    "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    # InternVL variants
    "internvl2.5-8b": "OpenGVLab/InternVL2_5-8B",
    # Ovis variants
    "ovis-9b": "AIDC-AI/Ovis2.5-9B",
    "ovis2.5-9b": "AIDC-AI/Ovis2.5-9B",
    "ovis2-8b": "AIDC-AI/Ovis2-8B",
    # LLaVA-Video variants
    "llava-video-7b": "lmms-lab/LLaVA-Video-7B-Qwen2",
    # VideoLLaMA variants
    "videollama-7b": "DAMO-NLP-SG/VideoLLaMA3-7B",
    "videollama3-7b": "DAMO-NLP-SG/VideoLLaMA3-7B",
    # Kimi-VL variants (Moonshot AI)
    "kimi-3b": "moonshotai/Kimi-VL-A3B-Instruct",
    "kimi-3b-thinking": "moonshotai/Kimi-VL-A3B-Thinking",
    # Generic video models (using generic loader)
    "videochat-7b": "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B",
    "oryx-7b": "THUdyh/Oryx-7B",
    "valley-7b": "bytedance-research/Valley-Eagle-7B",
    "video-r1-7b": "Video-R1/Video-R1-7B",
    "lumian-7b": "prithivMLmods/Lumian-VLR-7B-Thinking",
    "hunyuan-7b": "TencentARC/ARC-Hunyuan-Video-7B",
    "internvideo-8b": "OpenGVLab/InternVideo2_5_Chat_8B",
}


def _get_loader_registry() -> list[tuple[list[str], str, str]]:
    """
    Return the loader registry as a list of (patterns, module_path, class_name) tuples.

    Patterns are checked against the lowercase model_path string. The first
    match wins, so more specific patterns should come before generic ones.

    Lazy imports are used so that only the matched loader module is ever imported.
    """
    return [
        # (substrings to match, module path relative to this package, class name)
        # More specific patterns first - check these before generic ones
        (["llava-video", "llava_video"],  ".llava_video",    "LLaVAVideoLoader"),
        (["videollama"],                   ".videollama",     "VideoLLaMALoader"),
        (["videochat", "oryx", "valley", "video-r1", "lumian", "hunyuan"],
                                           ".video_generic",  "GenericVideoLoader"),
        (["kimi"],                         ".kimi",           "KimiVLLoader"),
        (["internvl", "internvideo"],      ".internvl",       "InternVLLoader"),
        (["nvila"],                        ".nvila",          "NVILALoader"),
        (["ovis"],                         ".ovis",           "OvisLoader"),
        # Generic patterns last (catch-all for base model families)
        (["qwen"],                         ".qwen_vl",        "QwenVLLoader"),
        (["llama"],                        ".llama_vision",   "LlamaVisionLoader"),
    ]


def get_loader_class(model_path: str) -> type["BaseVLMLoader"]:
    """
    Return the loader class for a given HuggingFace model path.

    Detection is based on substring matching against the model path.
    Raises ValueError if no matching loader is found.
    """
    model_path_lower = model_path.lower()

    for patterns, module_path, class_name in _get_loader_registry():
        if any(p in model_path_lower for p in patterns):
            import importlib
            module = importlib.import_module(module_path, package=__package__)
            return getattr(module, class_name)

    raise ValueError(
        f"No loader found for model: {model_path}\n"
        f"Supported model families: {list_supported_families()}\n"
        f"Available shortcuts: {list(MODEL_SHORTCUTS.keys())}"
    )


def resolve_model_path(model_name: str) -> str:
    """
    Resolve a model shortcut or full path to a HuggingFace model path.

    If model_name is a known shortcut (e.g. 'qwen-7b'), returns the full
    HuggingFace path. Lookup is case-insensitive. If no shortcut matches,
    returns the input unchanged, assuming it is already a full HuggingFace path.
    """
    return MODEL_SHORTCUTS.get(model_name) or MODEL_SHORTCUTS.get(model_name.lower(), model_name)


def list_supported_families() -> list[str]:
    """Return list of supported model family keywords."""
    families = []
    for patterns, _, _ in _get_loader_registry():
        families.extend(patterns)
    return families


def list_supported_models() -> dict[str, str]:
    """Return all available model shortcuts and their full paths."""
    return dict(MODEL_SHORTCUTS)


def register_model(shortcut: str, model_path: str) -> None:
    """
    Register a new model shortcut at runtime.

    This is useful for adding custom or fine-tuned model paths
    without modifying the source code.
    """
    MODEL_SHORTCUTS[shortcut] = model_path
