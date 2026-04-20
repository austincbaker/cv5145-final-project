"""Multimodal video dataset for InternVL2.5 fine-tuning.

Every sample returns a dict consumable by InternVLChatModel.forward:

    {
      "pixel_values":    FloatTensor[F, 3, 448, 448],   # F frames
      "input_ids":       LongTensor[L],
      "attention_mask":  LongTensor[L],
      "image_flags":     LongTensor[F],                 # 1 per usable frame
      "labels":          LongTensor[L],                 # -100 outside the answer span
    }

Prompt layout (internvl2_5 conversation template):

    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    Frame 1: <image>
    Frame 2: <image>
    ...
    Frame F: <image>
    {video_context_as_question_guidance}
    Question: {prompt}<|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>

Each `<image>` is expanded into `<img>` + `<IMG_CONTEXT>` * num_image_token + `</img>`
so the model sees a fixed number of visual tokens per frame at the correct positions.

Label masking: we tokenize the prompt-only prefix and the full sequence separately
and mask everything before `len(prompt_ids)` to -100 so loss is computed only on
the assistant's answer tokens (standard SFT with chat template).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_frames(frames_dir: Path, video_name: str, n_frames: int,
                 transform: transforms.Compose) -> torch.Tensor:
    """Load n_frames jpgs from {frames_dir}/{stem}/frame_{i:02d}.jpg."""
    stem = Path(video_name).stem
    video_frames_dir = frames_dir / stem
    pixels = []
    for i in range(n_frames):
        img_path = video_frames_dir / f"frame_{i:02d}.jpg"
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            pixels.append(transform(img))
    return torch.stack(pixels)  # [F, 3, H, W]


def _expand_image_tokens(text: str, num_patches_list: list[int],
                         num_image_token: int) -> str:
    """Replace each '<image>' placeholder with the IMG_CONTEXT expansion.

    Mirrors InternVLChatModel.chat() preprocessing so the visual token count
    matches what `extract_feature` produces.
    """
    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * num_image_token * num_patches
            + IMG_END_TOKEN
        )
        text = text.replace("<image>", image_tokens, 1)
    return text


def format_chat_prompt(system_message: str, user_content: str,
                       assistant_content: str | None,
                       role_user: str = "<|im_start|>user\n",
                       role_assistant: str = "<|im_start|>assistant\n",
                       sep: str = "<|im_end|>\n") -> str:
    """Apply the internvl2_5 chat template.

    If assistant_content is None the prompt ends ready for generation;
    otherwise it's a full training example including the target answer.
    """
    system = f"<|im_start|>system\n{system_message}{sep}" if system_message else ""
    user = f"{role_user}{user_content}{sep}"
    if assistant_content is None:
        assistant = role_assistant  # generation-ready; model writes its answer
    else:
        assistant = f"{role_assistant}{assistant_content}{sep}"
    return system + user + assistant


def build_user_content(frames_per_video: int, question: str) -> str:
    """The user-turn body, with one '<image>' per frame.

    Intentionally does NOT include the `video_context` annotation string. The
    model must recover aggressor/victim/action from the frames themselves;
    feeding the annotator's description alongside the frames would let the
    model short-circuit the vision path. Placeholder expansion happens in a
    separate step so we don't have to know num_image_token here.
    """
    frame_lines = "\n".join(f"Frame {i+1}: <image>" for i in range(frames_per_video))
    return f"{frame_lines}\n\nQuestion: {question}"


@dataclass
class MultimodalExample:
    pixel_values: torch.Tensor       # [F, 3, H, W]
    input_ids: torch.Tensor          # [L]
    attention_mask: torch.Tensor     # [L]
    image_flags: torch.Tensor        # [F]
    labels: torch.Tensor             # [L]


class VideoSFTDataset(Dataset):
    """Dataset for Phase 1 SFT and Phase 3 CoT SFT.

    Each example is expected to have:
        video_name, video_context, prompt, correct_answer
    Optional:
        reasoning (prepended to answer for CoT training).
    """

    def __init__(self, data_path: str, tokenizer, config: dict,
                 num_image_token: int, system_message: str = ""):
        with open(data_path) as f:
            raw = json.load(f)
        # Allow the CoT chains file format where each item may wrap answer+reasoning.
        self.examples = [self._normalize(ex) for ex in raw]
        self.tokenizer = tokenizer
        self.cfg = config
        self.num_image_token = num_image_token
        self.system_message = system_message
        self.transform = build_image_transform(config["video"]["image_size"])
        self.frames_dir = Path(config["video"]["frames_dir"])
        self.n_frames = int(config["video"]["frames_per_video"])
        self.num_patches = int(config["video"].get("num_patches_per_frame", 1))
        self.max_length = int(config["tokenization"]["max_length"])

    @staticmethod
    def _normalize(ex: dict) -> dict:
        # Supports raw SFT example keys as well as the CoT chains format
        # where each entry has nested {prompt, chosen:{answer, reasoning}, ...}.
        if "chosen" in ex and isinstance(ex["chosen"], dict):
            chosen = ex["chosen"]
            return {
                "video_name": ex.get("video_name", ""),
                "video_context": ex.get("video_context", ""),
                "prompt": ex.get("prompt", ""),
                "correct_answer": chosen.get("answer", ""),
                "reasoning": chosen.get("reasoning", ""),
            }
        return {
            "video_name": ex.get("video_name", ""),
            "video_context": ex.get("video_context", ""),
            "prompt": ex.get("prompt", ""),
            "correct_answer": ex.get("correct_answer", ""),
            "reasoning": ex.get("reasoning", ""),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def _build_answer(self, ex: dict) -> str:
        if ex.get("reasoning"):
            return f"Reasoning:\n{ex['reasoning']}\n\nAnswer: {ex['correct_answer']}"
        return ex["correct_answer"]

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        pixel_values = _load_frames(self.frames_dir, ex["video_name"],
                                    self.n_frames, self.transform)

        user_content = build_user_content(self.n_frames, ex["prompt"])
        # Expand image tokens to match pixel_values (one patch per frame by default).
        num_patches_list = [self.num_patches] * self.n_frames
        user_content_expanded = _expand_image_tokens(
            user_content, num_patches_list, self.num_image_token,
        )

        prompt_text = format_chat_prompt(self.system_message, user_content_expanded,
                                         assistant_content=None)
        full_text = format_chat_prompt(self.system_message, user_content_expanded,
                                       assistant_content=self._build_answer(ex))

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False,
                                    return_tensors="pt")["input_ids"][0]
        full_enc = self.tokenizer(full_text, add_special_tokens=False,
                                  max_length=self.max_length, truncation=True,
                                  padding="max_length", return_tensors="pt")
        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]

        # Mask prompt tokens (and padding) from the loss.
        labels = input_ids.clone()
        prompt_len = min(prompt_ids.numel(), input_ids.numel())
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        image_flags = torch.ones(self.n_frames * self.num_patches, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
            "labels": labels,
        }


class VideoDPOPairDataset(Dataset):
    """Dataset for Phase 5 ADPO preference pairs.

    Each sample returns two full InternVL-ready sequences (chosen, rejected)
    plus the shared pixel_values.
    """

    def __init__(self, pairs_path: str, tokenizer, config: dict,
                 num_image_token: int, system_message: str = ""):
        with open(pairs_path) as f:
            raw = json.load(f)
        self.pairs = [p for p in raw if p.get("rejected")]
        self.tokenizer = tokenizer
        self.cfg = config
        self.num_image_token = num_image_token
        self.system_message = system_message
        self.transform = build_image_transform(config["video"]["image_size"])
        self.frames_dir = Path(config["video"]["frames_dir"])
        self.n_frames = int(config["video"]["frames_per_video"])
        self.num_patches = int(config["video"].get("num_patches_per_frame", 1))
        self.max_length = int(config["tokenization"]["max_length"])

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode_response(self, pair: dict, answer_text: str) -> dict:
        user_content = build_user_content(self.n_frames, pair["prompt"])
        num_patches_list = [self.num_patches] * self.n_frames
        user_content_expanded = _expand_image_tokens(
            user_content, num_patches_list, self.num_image_token,
        )
        prompt_text = format_chat_prompt(self.system_message, user_content_expanded,
                                         assistant_content=None)
        full_text = format_chat_prompt(self.system_message, user_content_expanded,
                                       assistant_content=answer_text)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False,
                                    return_tensors="pt")["input_ids"][0]
        full_enc = self.tokenizer(full_text, add_special_tokens=False,
                                  max_length=self.max_length, truncation=True,
                                  padding="max_length", return_tensors="pt")
        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]

        # response_mask: 1 on answer tokens (post-prompt, non-padding), 0 elsewhere.
        response_mask = torch.zeros_like(input_ids, dtype=torch.long)
        prompt_len = min(prompt_ids.numel(), input_ids.numel())
        response_mask[prompt_len:] = 1
        response_mask = response_mask * attention_mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        pixel_values = _load_frames(self.frames_dir, pair["video_name"],
                                    self.n_frames, self.transform)

        chosen_ans = pair["chosen"]["answer"]
        if pair["chosen"].get("reasoning"):
            chosen_ans = f"Reasoning:\n{pair['chosen']['reasoning']}\n\nAnswer: {chosen_ans}"
        rejected_ans = pair["rejected"][0]["answer"]

        chosen = self._encode_response(pair, chosen_ans)
        rejected = self._encode_response(pair, rejected_ans)
        image_flags = torch.ones(self.n_frames * self.num_patches, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "chosen_input_ids":      chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_response_mask":  chosen["response_mask"],
            "rejected_input_ids":      rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_response_mask":  rejected["response_mask"],
            "image_flags": image_flags,
        }


def register_image_context_token(tokenizer) -> int:
    """Ensure `<IMG_CONTEXT>` is a single, fixed token id in the tokenizer.

    InternVL's tokenizer includes this token already; we just fetch and return
    its id so the training loop can plug it into `model.img_context_token_id`.
    """
    token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if token_id is None or token_id == tokenizer.unk_token_id:
        # Fallback: add it (shouldn't happen with InternVL2.5's tokenizer).
        tokenizer.add_special_tokens({"additional_special_tokens": [IMG_CONTEXT_TOKEN]})
        token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return int(token_id)


def collate_sft(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack per-sample tensors; concatenate pixel_values/image_flags along axis 0."""
    return {
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image_flags": torch.cat([b["image_flags"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def collate_dpo(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_flags": torch.cat([b["image_flags"] for b in batch], dim=0),
        "chosen_input_ids":      torch.stack([b["chosen_input_ids"] for b in batch]),
        "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch]),
        "chosen_response_mask":  torch.stack([b["chosen_response_mask"] for b in batch]),
        "rejected_input_ids":      torch.stack([b["rejected_input_ids"] for b in batch]),
        "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch]),
        "rejected_response_mask":  torch.stack([b["rejected_response_mask"] for b in batch]),
    }
