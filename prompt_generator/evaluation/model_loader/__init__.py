"""
Unified Vision-Language Model Loader

A generalized loader supporting multiple VLM families with a consistent interface.

Supported Models:
- Qwen2.5-VL (Alibaba) - 3B / 7B / 72B
- InternVL3 (OpenGVLab) - 2B / 8B / 78B
- Llama 3.2 Vision (Meta) - 11B / 90B
- NVILA (NVIDIA) - 8B / 15B
- Ovis 2.5 (AIDC-AI) - 2B / 9B

Usage:
    from model_loader import create_loader, ModelConfig

    config = ModelConfig(
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        max_new_tokens=64,
    )

    loader = create_loader(config)
    loader.load()
    response = loader.generate_response(frames, "What is happening?")

CLI shortcut usage:
    from model_loader import create_loader_from_shortcut

    loader = create_loader_from_shortcut("qwen-72b")
"""
from .base import BaseVLMLoader, ModelConfig
from .registry import (
    get_loader_class,
    list_supported_models,
    list_supported_families,
    register_model,
    resolve_model_path,
    MODEL_SHORTCUTS,
)

__all__ = [
    # Core classes
    "BaseVLMLoader",
    "ModelConfig",
    # Factory functions
    "create_loader",
    "create_loader_from_shortcut",
    "load_model",
    # Registry
    "get_loader_class",
    "list_supported_models",
    "list_supported_families",
    "register_model",
    "resolve_model_path",
    "MODEL_SHORTCUTS",
]

__version__ = "0.2.0"


def create_loader(config: ModelConfig | str) -> BaseVLMLoader:
    """
    Create the appropriate loader for a model.

    Automatically detects the model family from the model path and
    returns an instance of the correct loader class.

    Args:
        config: ModelConfig instance or full HuggingFace model path string

    Returns:
        Initialized (but not loaded) loader instance

    Example:
        >>> loader = create_loader("Qwen/Qwen2.5-VL-72B-Instruct")
        >>> loader.load()
        >>> response = loader.generate_response(images, "Describe this.")
    """
    if isinstance(config, str):
        config = ModelConfig(model_path=config)

    loader_class = get_loader_class(config.model_path)
    return loader_class(config)


def create_loader_from_shortcut(
    shortcut: str,
    **config_overrides,
) -> BaseVLMLoader:
    """
    Create a loader using a model shortcut name.

    Resolves the shortcut to a full HuggingFace path, then creates
    the appropriate loader with optional config overrides.

    Args:
        shortcut: Model shortcut (e.g. 'qwen-72b', 'llama-90b')
        **config_overrides: Override any ModelConfig fields

    Returns:
        Initialized (but not loaded) loader instance

    Example:
        >>> loader = create_loader_from_shortcut("qwen-72b", device_map="auto")
        >>> loader.load()
    """
    model_path = resolve_model_path(shortcut)
    config = ModelConfig(model_path=model_path, **config_overrides)
    return create_loader(config)


def load_model(
    model_path: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    device_map: str | None = None,
    **kwargs,
) -> BaseVLMLoader:
    """
    Convenience function to create, configure, and load a model in one call.

    Accepts both full HuggingFace paths and shortcut names.

    Args:
        model_path: HuggingFace model path or shortcut name
        dtype: Data type ("float16", "bfloat16", "float32")
        device: Device to load on ("cuda", "cpu")
        device_map: Device map for multi-GPU ("auto", "balanced", etc.)
        **kwargs: Additional ModelConfig parameters

    Returns:
        Loaded and ready-to-use loader instance

    Example:
        >>> loader = load_model("qwen-72b", device_map="auto")
        >>> response = loader.generate_response(images, "What do you see?")
    """
    resolved_path = resolve_model_path(model_path)
    config = ModelConfig(
        model_path=resolved_path,
        dtype=dtype,
        device=device,
        device_map=device_map,
        **kwargs,
    )

    loader = create_loader(config)
    loader.load()

    return loader
