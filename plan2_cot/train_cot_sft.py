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
from transformers import AutoProcessor, AutoModel, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np


class CoTDataset(Dataset):
    """Dataset supporting both direct answers and reasoning chains."""

    def __init__(self, data_path: str, processor, max_length: int = 2048):
        with open(data_path) as f:
            self.examples = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Build prompt: video context + question
        prompt = f"{ex['video_context']}\n\nQuestion: {ex['prompt']}\n\n"

        # Build answer based on whether CoT is available
        if ex.get("used_cot") and "reasoning_chain" in ex:
            # CoT format: reasoning → answer
            reasoning = ex["reasoning_chain"]
            answer = ex["correct_answer"]
            text = f"{prompt}Reasoning:\n{reasoning}\n\nAnswer: {answer}"
        else:
            # Direct format: just answer
            answer = ex["correct_answer"]
            text = f"{prompt}Answer: {answer}"

        # Tokenize
        encoding = self.processor(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove batch dimension
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "video_name": ex["video_name"],
            "question_type": ex["question_type"],
            "correct_answer": answer,
            "prompt": ex["prompt"],
            "used_cot": ex.get("used_cot", False),
        }


def train_cot_sft(
    train_data: str = "plan2_cot/cot_chains_train.json",
    val_data: str = "plan2_data/sft_val.json",
    checkpoint_path: str = "plan2_models/sft_baseline",
    output_dir: str = "plan2_models/cot_sft",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 8,
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

    # Load base model
    print(f"Loading base model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

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
    train_dataset = CoTDataset(train_data, processor)
    print(f"  {len(train_dataset)} examples")

    # Count CoT vs direct
    with open(train_data) as f:
        examples = json.load(f)
    cot_count = sum(1 for ex in examples if ex.get("used_cot"))
    print(f"  {cot_count} with CoT reasoning, {len(examples) - cot_count} direct")

    print(f"\nLoading validation data from {val_data}")
    val_dataset = CoTDataset(val_data, processor)
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
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        seed=seed,
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
