"""
Loader for InternVL vision-language models (OpenGVLab/Shanghai AI Lab).

Supports:
- OpenGVLab/InternVL2.5-1B through 78B
- OpenGVLab/InternVL3-1B through 78B
"""
from .base import BaseVLMLoader, ModelConfig


class InternVLLoader(BaseVLMLoader):
    """
    Loader for InternVL2.5 and InternVL3 models.
    
    Features:
    - Large 6B vision encoder (InternViT)
    - Strong multimodal reasoning
    - Variable Visual Position Encoding (V2PE)
    - Good video understanding (MVBench, MLVU)
    """
    
    MODEL_FAMILY = "internvl"
    
    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self.tokenizer = None
        self._generation_config = None
    
    def load(self) -> None:
        if self.model is not None:
            return
        
        torch = self._get_torch()
        self._setup_cuda_optimizations()
        
        from transformers import AutoModel, AutoTokenizer
        
        print(f"Loading InternVL model: {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        attn_impl = self._get_attention_implementation()
        print(f"Using attention: {attn_impl}")
        
        load_kwargs = {
            "torch_dtype": self._get_dtype(),
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }

        # AWQ-quantized InternVL variants (loaded via trust_remote_code) hit
        # a NotImplementedError "Cannot copy out of meta tensor" when the
        # default low_cpu_mem_usage=True path is combined with the manual
        # .to(device) call below. Force accelerate to do the placement via
        # a device_map so the AWQ modules get materialised correctly and
        # the manual .to() is skipped.
        #
        # Use {"": 0} (single-device pin) rather than "auto": accelerate's
        # "auto" planner splits InternVL's vision encoder and LLM across
        # CPU/GPU for large-but-fits-on-one-GPU AWQ models, which keeps
        # the load from erroring but produces grammatical-garbage outputs
        # because cross-modal attention runs across devices at mismatched
        # dtypes. Pinning to a single device matches what the Qwen2VL
        # loader already does implicitly (moves to self.config.device).
        is_awq_model = "awq" in self.config.model_path.lower()
        if self.config.device_map:
            effective_device_map = self.config.device_map
        elif is_awq_model:
            effective_device_map = {"": 0}
        else:
            effective_device_map = None
        if effective_device_map is not None:
            load_kwargs["device_map"] = effective_device_map

        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            **load_kwargs,
        )

        if not effective_device_map:
            self.model = self.model.to(self.config.device)

        self.model.eval()
        
        # Store generation config if available
        if hasattr(self.model, 'generation_config'):
            self._generation_config = self.model.generation_config
        
        if self.config.use_torch_compile:
            self._apply_torch_compile()
        
        print(f"InternVL model loaded. Memory: {self.get_memory_usage()['allocated_mb']:.0f}MB")
    
    def _load_images(self, images: list) -> "torch.Tensor":
        """
        Load and preprocess images for InternVL.
        
        InternVL uses dynamic resolution with pixel shuffle.
        """
        torch = self._get_torch()
        from torchvision import transforms
        
        # InternVL standard preprocessing
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        pixel_values = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            pixel_values.append(transform(img))
        
        return torch.stack(pixel_values).to(self.config.device, dtype=self._get_dtype())
    
    def _build_prompt(self, prompt: str, num_images: int) -> str:
        """
        Build InternVL chat prompt with image placeholders.
        """
        # InternVL uses <image> tags for image placeholders
        image_tags = "<image>\n" * num_images
        return f"{image_tags}{prompt}"
    
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

        # Text-only mode: no images
        if not images:
            if hasattr(self.model, 'chat'):
                response = self.model.chat(
                    self.tokenizer,
                    None,
                    prompt,
                    generation_config=dict(
                        max_new_tokens=max_new_tokens,
                        do_sample=self.config.do_sample,
                        temperature=self.config.temperature if self.config.do_sample else None,
                    ),
                )
                return response
            # Fallback: use chat template for proper formatting
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                text_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                text_prompt = prompt
            input_ids = self.tokenizer(
                text_prompt, return_tensors="pt",
            ).input_ids.to(self.config.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature if self.config.do_sample else None,
                )
            return self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:], skip_special_tokens=True,
            )

        # Load and preprocess images
        pixel_values = self._load_images(images)

        # Build prompt with image placeholders
        full_prompt = self._build_prompt(prompt, len(images))

        # Use model's chat method if available (preferred for InternVL)
        if hasattr(self.model, 'chat'):
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config=dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature if self.config.do_sample else None,
                ),
            )
            return response

        # Fallback to manual generation
        input_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
        ).input_ids.to(self.config.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response
    
    def generate_response_multi_turn(
        self,
        images: list,
        conversation: list[dict],
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """
        Multi-turn conversation with InternVL.
        
        Args:
            images: List of PIL images
            conversation: List of {"role": "user"|"assistant", "content": str}
            max_new_tokens: Max tokens to generate
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        torch = self._get_torch()
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        pixel_values = self._load_images(images)
        
        # Build multi-turn prompt
        prompt_parts = []
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")
        
        # Add image placeholders at the start
        image_tags = "<image>\n" * len(images)
        full_prompt = image_tags + "\n".join(prompt_parts) + "\nAssistant:"
        
        if hasattr(self.model, 'chat'):
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config=dict(max_new_tokens=max_new_tokens),
            )
            return response
        
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.config.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
            )
        
        return self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)