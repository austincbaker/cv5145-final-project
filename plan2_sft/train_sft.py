#!/usr/bin/env python3
"""
Phase 1: SFT Baseline Training.

Fine-tune InternVL2.5-8B on the benchmark questions using LoRA.
Input: video context + question
Output: correct answer
"""

import json
import argparse
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np


class SFTDataset(Dataset):
    """SFT dataset for video question answering (text-only Phase 1)."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        with open(data_path) as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = f"{ex['video_context']}\n\nQuestion: {ex['prompt']}"
        text = f"{prompt}\n\nAnswer: {ex['correct_answer']}"

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def setup_model(model_name: str = "OpenGVLab/InternVL2_5-8B", lora_rank: int = 8):
    """Load InternVL2.5-8B's language backbone and apply LoRA.

    Phase 1 is text-only, so we train just the InternLM2.5 language model
    submodule. This avoids InternVLChatModel.forward() which does not accept
    inputs_embeds (the path Trainer/PEFT uses).
    """
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = full_model.language_model
    print(f"Extracted language backbone: {type(model).__name__}")

    # Required for gradient checkpointing with PEFT — frozen base params
    # won't propagate grads back to inputs otherwise
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["wqkv", "wo", "w1", "w2", "w3"],
    )

    model = get_peft_model(model, lora_config)
    print(f"Applied LoRA with rank={lora_rank}")
    model.print_trainable_parameters()

    return tokenizer, model


def train(
    train_data: str = "plan2_data/sft_train.json",
    val_data: str = "plan2_data/sft_val.json",
    output_dir: str = "plan2_models/sft_baseline",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 5e-4,
    warmup_steps: int = 500,
    logging_steps: int = 50,
    eval_steps: int = 500,
    save_steps: int = 500,
    seed: int = 42,
    force: bool = False,
):
    """Train SFT baseline on video QA."""
    out = Path(output_dir)
    has_adapter = (out / "adapter_config.json").exists() or any(out.glob("checkpoint-*/adapter_config.json"))
    if not force and has_adapter:
        print(f"Skipping training: checkpoint already exists at {output_dir} (use --force to retrain)")
        return None

    transformers.set_seed(seed)
    torch.manual_seed(seed)

    print("=" * 80)
    print("PHASE 1: SFT BASELINE TRAINING")
    print("=" * 80)

    # Setup
    tokenizer, model = setup_model(model_name)

    # Load datasets
    print(f"Loading training data from {train_data}")
    train_dataset = SFTDataset(train_data, tokenizer)
    print(f"  {len(train_dataset)} examples")

    print(f"Loading validation data from {val_data}")
    val_dataset = SFTDataset(val_data, tokenizer)
    print(f"  {len(val_dataset)} examples")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        seed=seed,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save the best LoRA adapter directly to output_dir so eval can find it
    print(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining complete. Best model saved to {output_dir}")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="plan2_data/sft_train.json")
    parser.add_argument("--val-data", default="plan2_data/sft_val.json")
    parser.add_argument("--output-dir", default="plan2_models/sft_baseline")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if checkpoint already exists")
    args = parser.parse_args()

    train(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        force=args.force,
    )
