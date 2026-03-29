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
    # LLaVA-Video-7B-Qwen2 uses LlavaQwenForCausalLM, which is only available
    # via the lmms-lab llava package (not in standard transformers).
    # Tuple form: (install_spec, import_name_to_check) — skips install if
    # the 'llava' module is already present (e.g. pre-installed on login node).
    EXTRA_PACKAGES = [("git+https://github.com/LLaVA-VL/LLaVA-NeXT.git", "llava")]

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
            use_fast=True,
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

        # AutoModelForCausalLM won't dispatch to LlavaQwenForCausalLM via the
        # standard LlavaConfig auto-registry. Import the class directly from the
        # lmms-lab llava package so we can call from_pretrained on it explicitly.
        try:
            from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
            model_cls = LlavaQwenForCausalLM
            print("Using LlavaQwenForCausalLM from llava package")
        except ImportError:
            model_cls = AutoModelForCausalLM
            print("Warning: llava package not available, falling back to AutoModelForCausalLM")

        self.model = model_cls.from_pretrained(
            self.config.model_path,
            **load_kwargs,
        )

        if not self.config.device_map:
            self.model = self.model.to(self.config.device)

        # Sync processor settings from the model's vision config so image sizes
        # and token counts match what the model actually expects.
        vision_cfg = getattr(self.model.config, "vision_config", None)
        patch_size = getattr(vision_cfg, "patch_size", None) if vision_cfg else None
        image_size = getattr(vision_cfg, "image_size", None) if vision_cfg else None

        # LlavaQwen may also expose mm_vision_tower config; try the vision tower directly.
        if patch_size is None:
            try:
                vt = self.model.get_vision_tower()
                patch_size = getattr(getattr(vt, "config", None), "patch_size", None)
                if image_size is None:
                    image_size = getattr(getattr(vt, "config", None), "image_size", None)
            except Exception:
                pass

        # Final fallback: LLaVA-Video uses CLIP-ViT-L/14@336 (patch=14, size=336).
        if patch_size is None:
            patch_size = 14
            print("Warning: patch_size not found in model config, defaulting to 14")
        if image_size is None:
            image_size = 336
            print("Warning: image_size not found in model config, defaulting to 336")

        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = patch_size
        if hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            ip.size = {"height": image_size, "width": image_size}
            ip.crop_size = {"height": image_size, "width": image_size}

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

        # Text-only mode: no images
        if not images:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                text_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                text_prompt = f"USER: {prompt}\nASSISTANT:"
            inputs = tokenizer(text_prompt, return_tensors="pt").to(self.config.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            with torch.inference_mode():
                # For text-only, bypass LlavaQwenForCausalLM's custom
                # generate() — it converts input_ids to inputs_embeds and
                # calls super().generate() without input_ids, which causes
                # immediate EOS. Call Qwen2ForCausalLM.generate() directly
                # so the standard HF generation loop keeps input_ids intact.
                from transformers import Qwen2ForCausalLM
                outputs = Qwen2ForCausalLM.generate(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            return tokenizer.decode(
                outputs[0][input_ids.shape[1]:], skip_special_tokens=True,
            ).strip()

        # Build prompt: one <image> token per frame (standard LLaVA format)
        image_tokens = "<image>" * len(images)
        content = f"{image_tokens}\n{prompt}"

        # Use the correct chat template for the model backbone.
        # LLaVA-Video-7B-Qwen2 requires the Qwen2 chat format; the old
        # Vicuna "USER: ... ASSISTANT:" format causes immediate EOS.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": content}]
            text_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text_prompt = f"USER: {content}\nASSISTANT:"

        # Process inputs
        inputs = self.processor(
            text=text_prompt,
            images=images,
            return_tensors="pt",
        ).to(self.config.device)

        # LlavaQwenForCausalLM.generate() signature is generate(self, inputs=None, images=None, ...)
        # Unpacking **inputs puts input_ids into **kwargs where it's ignored, leaving inputs=None.
        # Pass input_ids positionally and map pixel_values -> images.
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        attention_mask = inputs.get("attention_mask")
        # Cast pixel_values to match model dtype (processor returns float32,
        # model mm_projector expects bfloat16/float16).
        if pixel_values is not None:
            model_dtype = next(self.model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=model_dtype)
        # image_sizes as (height, width) tuples for the token count calculation
        image_sizes = [(img.size[1], img.size[0]) for img in images]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                use_cache=True,
                **kwargs,
            )

        # Decode only the newly generated tokens.
        # LlavaQwenForCausalLM.generate() returns ONLY generated tokens,
        # unlike standard HuggingFace generate() which returns input+output.
        # Handle both cases.
        if output_ids.shape[1] > input_ids.shape[1]:
            generated_ids = output_ids[0][input_ids.shape[1]:]
        else:
            generated_ids = output_ids[0]

        response = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return response
