"""
Loader for LLaVA-Video models.

Supports:
- LLaVA-Video-7B-Qwen2
- Other LLaVA-Video variants
"""
from .base import BaseVLMLoader, ModelConfig


class LLaVAVideoLoader(BaseVLMLoader):
    """
    Loader for LLaVA-Video models.

    LLaVA-Video is designed for video understanding with efficient
    frame encoding and cross-modal attention.
    """

    MODEL_FAMILY = "llava-video"

    def load(self) -> None:
        if self.model is not None:
            return

        torch = self._get_torch()
        self._setup_cuda_optimizations()

        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading LLaVA-Video model: {self.config.model_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
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

        print(f"LLaVA-Video loaded. Memory: {self.get_memory_usage()['allocated_mb']:.0f}MB")

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

        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=images,
            return_tensors="pt",
        ).to(self.config.device)

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                use_cache=True,
                **kwargs,
            )

        # Decode
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return response
