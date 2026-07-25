"""
Loader for Gemma 4 vision-language models.

Requires transformers >= 5.5.0 and the vlm_gemma4 conda env.
"""
from .base import BaseVLMLoader, ModelConfig


class GemmaVLLoader(BaseVLMLoader):

    MODEL_FAMILY = "gemma"
    EXTRA_PACKAGES = []

    def load(self) -> None:
        if self.model is not None:
            return

        torch = self._get_torch()
        self._setup_cuda_optimizations()

        from transformers import AutoProcessor, AutoModelForImageTextToText

        print("Loading Gemma model: %s" % self.config.model_path)

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )

        attn_impl = "sdpa"
        print("Using attention: %s" % attn_impl)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path,
            torch_dtype=self._get_dtype(),
            device_map="auto",
            attn_implementation=attn_impl,
        )
        self.model.eval()

        print("Model loaded. Memory: %.0fMB" % self.get_memory_usage()["allocated_mb"])

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

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
            )

        response = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        return response
