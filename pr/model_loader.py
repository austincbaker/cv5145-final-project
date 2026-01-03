"""
Optimized model loader with batching, torch.compile, and flash attention support.
"""
from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class ModelConfig:
    model_path: str = "AIDC-AI/Ovis2.5-9B"
    enable_thinking: bool = True
    enable_thinking_budget: bool = True
    thinking_budget: int = 256  # Reduced from 512 for faster inference
    max_new_tokens: int = 128   # Reduced from 1024 - answers are short
    image_size: tuple[int, int] = (224, 224)
    dtype: str = "float16"
    device: str = "cuda"
    
    # Optimization flags
    use_torch_compile: bool = True
    use_flash_attention: bool = True
    compile_mode: str = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"
    
    # Batch inference settings
    enable_batching: bool = True
    max_batch_size: int = 4


class OvisModelLoader:
    """
    Optimized model loader for Ovis vision-language models.
    
    Features:
    - torch.compile for faster inference (PyTorch 2.0+)
    - Flash Attention 2 support
    - Batch inference for multiple prompts
    - Memory-efficient loading
    """
    
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.processor = None
        self._torch = None
        self._compiled = False

    def _get_torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _setup_cuda_optimizations(self) -> None:
        torch = self._get_torch()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    def _get_dtype(self):
        torch = self._get_torch()
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float16)

    def _get_attention_implementation(self) -> str:
        """
        Determine the best attention implementation.
        - Flash Attention 2: Best (if installed).
        - SDPA: Fast default for PyTorch 2.0+ (standard for Llama, Mistral, etc).
        - Eager: Slow fallback (required for Ovis 2.5 if FA2 is missing).
        """
        # 1. Obey manual config override
        if not self.config.use_flash_attention:
            return "eager"
        
        # 2. Try to use Flash Attention 2 (Preferred for everything)
        try:
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            print("flash_attn module not found. is it installed?")
            pass
        
        # 3. INTELLIGENT FALLBACK
        # Check if the model path string contains "ovis"
        is_ovis = "ovis" in self.config.model_path.lower()
        
        if is_ovis:
            # Ovis 2.5 crashes with SDPA, so we MUST use Eager
            print(f"Detected Ovis model without Flash Attention. Falling back to 'eager' to prevent crash.")
            return "eager"
        
        # 4. For all other models (Llama, Qwen, etc.), use SDPA
        # It is much faster than eager and uses less memory.
        return "sdpa"

    def load(self) -> None:
        if self.model is not None:
            return

        torch = self._get_torch()
        self._setup_cuda_optimizations()

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        attn_impl = self._get_attention_implementation()
        print(f"Using attention implementation: {attn_impl}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self._get_dtype(),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        ).to(self.config.device)

        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        self.model.config.use_cache = True
        
        # Apply torch.compile for PyTorch 2.0+
        if self.config.use_torch_compile and not self._compiled:
            self._apply_torch_compile()

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the model for faster inference."""
        torch = self._get_torch()
        
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile not available (requires PyTorch 2.0+)")
            return
        
        try:
            # Compile the model
            self.model = torch.compile(
                self.model, 
                mode=self.config.compile_mode,
                fullgraph=False,  # Allow graph breaks for flexibility
            )
            self._compiled = True
            print(f"Model compiled with mode: {self.config.compile_mode}")
        except Exception as e:
            warnings.warn(f"torch.compile failed: {e}. Continuing without compilation.")

    def generate_response(self, images: list, prompt: str) -> str:
        """Generate a single response for one prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        torch = self._get_torch()

        resized_images = [
            img.resize(self.config.image_size) for img in images
        ]

        content = []
        for img in resized_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

        input_ids = input_ids.to(self.config.device)
        pixel_values = pixel_values.half().to(self.config.device)
        grid_thws = grid_thws.to(self.config.device)

        with torch.cuda.amp.autocast(dtype=self._get_dtype()):
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=self.config.enable_thinking,
                enable_thinking_budget=self.config.enable_thinking_budget,
                max_new_tokens=self.config.max_new_tokens,
                thinking_budget=self.config.thinking_budget,
                output_attentions=False,
                output_hidden_states=False,
            )

        response = self.model.text_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        return response

    def generate_responses_batch(
        self, 
        images: list, 
        prompts: list[str]
    ) -> list[str]:
        """
        Generate responses for multiple prompts using the same images.
        
        This is more efficient than calling generate_response multiple times
        because it reuses the image encoding.
        
        Args:
            images: List of PIL images (same for all prompts)
            prompts: List of text prompts
            
        Returns:
            List of response strings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not self.config.enable_batching:
            # Fall back to sequential processing
            return [self.generate_response(images, p) for p in prompts]
        
        torch = self._get_torch()
        
        resized_images = [
            img.resize(self.config.image_size) for img in images
        ]
        
        responses = []
        
        # Process in batches
        for i in range(0, len(prompts), self.config.max_batch_size):
            batch_prompts = prompts[i:i + self.config.max_batch_size]
            batch_responses = self._process_prompt_batch(resized_images, batch_prompts)
            responses.extend(batch_responses)
        
        return responses

    def _process_prompt_batch(
        self, 
        images: list, 
        prompts: list[str]
    ) -> list[str]:
        """Process a batch of prompts with shared images."""
        torch = self._get_torch()
        
        # For models that don't support true batching, process sequentially
        # but with shared image preprocessing
        responses = []
        
        # Preprocess images once
        content_base = [{"type": "image", "image": img} for img in images]
        
        for prompt in prompts:
            content = content_base.copy()
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )
            
            input_ids = input_ids.to(self.config.device)
            pixel_values = pixel_values.half().to(self.config.device)
            grid_thws = grid_thws.to(self.config.device)
            
            with torch.cuda.amp.autocast(dtype=self._get_dtype()):
                outputs = self.model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    enable_thinking=self.config.enable_thinking,
                    enable_thinking_budget=self.config.enable_thinking_budget,
                    max_new_tokens=self.config.max_new_tokens,
                    thinking_budget=self.config.thinking_budget,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            
            response = self.model.text_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            responses.append(response)
        
        return responses

    def warmup(self, sample_image_count: int = 4) -> None:
        """
        Warmup the model with dummy inputs to trigger torch.compile.
        
        Call this after load() to ensure the first real inference isn't slow.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        torch = self._get_torch()
        from PIL import Image
        import numpy as np
        
        print("Warming up model...")
        
        # Create dummy images
        dummy_images = [
            Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            for _ in range(sample_image_count)
        ]
        
        dummy_prompt = "What is shown in this video?"
        
        # Run warmup inference
        with torch.no_grad():
            _ = self.generate_response(dummy_images, dummy_prompt)
        
        print("Warmup complete.")

    def unload(self) -> None:
        if self.model is not None:
            torch = self._get_torch()
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._compiled = False
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        torch = self._get_torch()
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }