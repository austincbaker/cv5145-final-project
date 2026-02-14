"""
Loader for VideoLLaMA models.

Supports:
- VideoLLaMA3-7B
- Other VideoLLaMA variants
"""
from .base import BaseVLMLoader, ModelConfig


class VideoLLaMALoader(BaseVLMLoader):
    """
    Loader for VideoLLaMA models.

    VideoLLaMA specializes in long-form video understanding with
    efficient temporal modeling.
    """

    MODEL_FAMILY = "videollama"

    def load(self) -> None:
        if self.model is not None:
            return

        torch = self._get_torch()
        self._setup_cuda_optimizations()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading VideoLLaMA model: {self.config.model_path}")

        # Load tokenizer
        self.processor = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Configure attention
        attn_impl = self._get_attention_implementation()
        print(f"Using attention: {attn_impl}")

        # Load model
        load_kwargs = {
            "torch_dtype": self._get_dtype(),
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "attn_implementation": attn_impl,
        }

        if self.config.device_map:
            load_kwargs["device_map"] = self.config.device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **load_kwargs,
        )

        if not self.config.device_map:
            self.model = self.model.to(self.config.device)

        self.model.eval()

        if self.config.use_torch_compile:
            self._apply_torch_compile()

        print(f"VideoLLaMA loaded. Memory: {self.get_memory_usage()['allocated_mb']:.0f}MB")

    def generate_response(
        self,
        images: list,
        prompt: str,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        torch = self._get_torch()
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # VideoLLaMA may have custom preprocessing
        # This is a generic implementation - may need adjustment
        try:
            # Try using model's native video processing if available
            if hasattr(self.model, 'generate_from_video'):
                response = self.model.generate_from_video(
                    frames=images,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
                return response
        except Exception:
            pass

        # Fallback: Standard image-based processing
        from transformers import AutoProcessor

        if not hasattr(self, '_image_processor'):
            self._image_processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
            )

        inputs = self._image_processor(
            text=prompt,
            images=images,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                **kwargs,
            )

        response = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

        return response
