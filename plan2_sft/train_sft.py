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
from transformers import AutoProcessor, AutoModel, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np


class SFTDataset(Dataset):
    """SFT dataset for video question answering."""

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
        prompt = f"{ex['video_context']}\n\nQuestion: {ex['prompt']}"
        answer = ex["correct_answer"]

        # Format as: prompt → answer
        text = f"{prompt}\n\nAnswer: {answer}"

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
        }


def setup_model(model_name: str = "OpenGVLab/InternVL2_5-8B", lora_rank: int = 8):
    """Load InternVL2.5-8B and apply LoRA."""
    print(f"Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # InternVL attention projections
    )

    model = get_peft_model(model, lora_config)
    print(f"Applied LoRA with rank={lora_rank}")
    model.print_trainable_parameters()

    return processor, model


def train(
    train_data: str = "plan2_data/sft_train.json",
    val_data: str = "plan2_data/sft_val.json",
    output_dir: str = "plan2_models/sft_baseline",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 5e-4,
    warmup_steps: int = 500,
    logging_steps: int = 50,
    eval_steps: int = 500,
    save_steps: int = 500,
    seed: int = 42,
):
    """Train SFT baseline on video QA."""
    transformers.set_seed(seed)
    torch.manual_seed(seed)

    print("=" * 80)
    print("PHASE 1: SFT BASELINE TRAINING")
    print("=" * 80)

    # Setup
    processor, model = setup_model(model_name)

    # Load datasets
    print(f"Loading training data from {train_data}")
    train_dataset = SFTDataset(train_data, processor)
    print(f"  {len(train_dataset)} examples")

    print(f"Loading validation data from {val_data}")
    val_dataset = SFTDataset(val_data, processor)
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
    parser.add_argument("--train-data", default="plan2_data/sft_train.json")
    parser.add_argument("--val-data", default="plan2_data/sft_val.json")
    parser.add_argument("--output-dir", default="plan2_models/sft_baseline")
    parser.add_argument("--model-name", default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
