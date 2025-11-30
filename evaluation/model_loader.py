from dataclasses import dataclass


@dataclass
class ModelConfig:
    # model_path: str = "AIDC-AI/Ovis2.5-2B"
    model_path: str = "AIDC-AI/Ovis2.5-9B"
    enable_thinking: bool = True
    enable_thinking_budget: bool = True
    thinking_budget: int = 512
    max_new_tokens: int = 1024
    image_size: tuple[int, int] = (224, 224)
    dtype: str = "float16"
    device: str = "cuda"


class OvisModelLoader:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.processor = None
        self._torch = None

    def _get_torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _setup_cuda_optimizations(self) -> None:
        torch = self._get_torch()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def _get_dtype(self):
        torch = self._get_torch()
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float16)

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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=self._get_dtype(),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.config.device)

        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        self.model.config.use_cache = True

    def generate_response(self, images: list, prompt: str) -> str:
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

    def unload(self) -> None:
        if self.model is not None:
            torch = self._get_torch()
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()