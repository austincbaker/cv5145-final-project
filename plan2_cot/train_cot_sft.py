#!/usr/bin/env python3
"""
Phase 3: CoT SFT Training.

Fine-tune Phase 1 checkpoint on mixed dataset:
  - Compound/complex questions: use reasoning chain → answer format
  - Simple questions: use direct answer format (no CoT overhead)

Resume from Phase 1 checkpoint to retain baseline knowledge.
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


class CoTDataset(Dataset):
    """Dataset supporting both direct answers and reasoning chains."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        with open(data_path) as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = f"{ex['video_context']}\n\nQuestion: {ex['prompt']}\n\n"

        if ex.get("used_cot") and "reasoning_chain" in ex:
            reasoning = ex["reasoning_chain"]
            answer = ex["correct_answer"]
            text = f"{prompt}Reasoning:\n{reasoning}\n\nAnswer: {answer}"
        else:
            answer = ex["correct_answer"]
            text = f"{prompt}Answer: {answer}"

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


def train_cot_sft(
    train_data: str = "plan2_cot/cot_chains_train.json",
    val_data: str = "plan2_data/sft_val.json",
    checkpoint_path: str = "plan2_models/sft_baseline",
    output_dir: str = "plan2_models/cot_sft",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 2e-4,
    warmup_steps: int = 200,
    logging_steps: int = 50,
    eval_steps: int = 500,
    save_steps: int = 500,
    seed: int = 42,
):
    """Train CoT SFT on mixed dataset, resuming from Phase 1 checkpoint."""
    transformers.set_seed(seed)
    torch.manual_seed(seed)

    print("=" * 80)
    print("PHASE 3: CoT SFT TRAINING")
    print("=" * 80)

    # Load base model (extract language backbone for text-only training)
    print(f"Loading base model: {model_name}")
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

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Load Phase 1 LoRA weights as checkpoint
    print(f"Loading Phase 1 checkpoint from {checkpoint_path}")
    from peft import PeftModel
    try:
        model = PeftModel.from_pretrained(model, checkpoint_path, device_map="auto")
        print("✓ Loaded Phase 1 LoRA checkpoint")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        print("Training from scratch instead")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["wqkv", "wo", "w1", "w2", "w3"],
        )
        model = get_peft_model(model, lora_config)

    print("Model ready for fine-tuning")

    # Load datasets
    print(f"\nLoading training data from {train_data}")
    train_dataset = CoTDataset(train_data, tokenizer)
    print(f"  {len(train_dataset)} examples")

    # Count CoT vs direct
    with open(train_data) as f:
        examples = json.load(f)
    cot_count = sum(1 for ex in examples if ex.get("used_cot"))
    print(f"  {cot_count} with CoT reasoning, {len(examples) - cot_count} direct")

    print(f"\nLoading validation data from {val_data}")
    val_dataset = CoTDataset(val_data, tokenizer)
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

    print(f"\nTraining complete. Best model saved to {output_dir}")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="plan2_cot/cot_chains_train.json")
    parser.add_argument("--val-data", default="plan2_data/sft_val.json")
    parser.add_argument("--checkpoint", default="plan2_models/sft_baseline")
    parser.add_argument("--output-dir", default="plan2_models/cot_sft")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_cot_sft(
        train_data=args.train_data,
        val_data=args.val_data,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
