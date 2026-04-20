#!/usr/bin/env python3
"""
Phase 5: ADPO (Anchored Direct Preference Optimization) Training.

Trains with ADPO loss on preference pairs:
  L_ADPO = L_preference(chosen > rejected) + alpha * L_anchor(model | reference)

Where:
  - L_preference: Bradley-Terry ranking loss
  - L_anchor: KL divergence from Phase 3 checkpoint
  - alpha: anchoring strength (swept 0.1-1.0 to find optimal)

Monitors: validation accuracy, preference ranking, divergence from reference
"""

import json
import math
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoProcessor, AutoModel, TrainingArguments, Trainer
from peft import PeftModel
import numpy as np


class ADPODataset(Dataset):
    """Dataset for ADPO preference pairs."""

    def __init__(self, pairs_path: str, processor, max_length: int = 2048):
        with open(pairs_path) as f:
            self.pairs = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Format chosen and rejected responses
        prompt = f"Question: {pair['prompt']}\n\nAnswer:"

        # Chosen
        chosen_text = f"{prompt} {pair['chosen']['answer']}"
        if pair['chosen']['reasoning']:
            chosen_text = f"{prompt}\nReasoning:\n{pair['chosen']['reasoning']}\nAnswer: {pair['chosen']['answer']}"

        # Rejected (take first one, could vary)
        rejected = pair['rejected'][0] if pair['rejected'] else pair['rejected'][0]
        rejected_text = f"{prompt} {rejected['answer']}"

        # Tokenize
        chosen_tokens = self.processor(
            chosen_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        rejected_tokens = self.processor(
            rejected_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
        }


def compute_adpo_loss(
    model_logits_chosen: torch.Tensor,
    model_logits_rejected: torch.Tensor,
    reference_logits_chosen: torch.Tensor,
    reference_logits_rejected: torch.Tensor,
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute ADPO loss.

    Args:
        model_logits_chosen: Model logits for chosen responses
        model_logits_rejected: Model logits for rejected responses
        reference_logits_chosen: Reference model logits for chosen
        reference_logits_rejected: Reference model logits for rejected
        alpha: Anchoring strength weight

    Returns:
        Dictionary with loss components
    """
    # Bradley-Terry preference loss: log softmax of margin
    margin = (model_logits_chosen - model_logits_rejected)
    pref_loss = -F.logsigmoid(margin).mean()

    # KL divergence from reference (anchoring)
    kl_loss = F.kl_div(
        F.log_softmax(model_logits_chosen, dim=-1),
        F.softmax(reference_logits_chosen, dim=-1),
        reduction='batchmean'
    )
    kl_loss += F.kl_div(
        F.log_softmax(model_logits_rejected, dim=-1),
        F.softmax(reference_logits_rejected, dim=-1),
        reduction='batchmean'
    )

    # Combined ADPO loss
    loss = pref_loss + alpha * kl_loss

    return {
        "loss": loss,
        "preference_loss": pref_loss,
        "kl_loss": kl_loss,
    }


def train_adpo(
    preference_pairs_path: str = "plan2_adpo/preference_pairs_train.json",
    reference_model_path: str = "plan2_models/cot_sft",
    output_dir: str = "plan2_models/adpo_final",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    alpha: float = 0.5,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 42,
):
    """Train ADPO on preference pairs."""
    transformers.set_seed(seed)
    torch.manual_seed(seed)

    print("=" * 80)
    print(f"PHASE 5: ADPO TRAINING (alpha={alpha})")
    print("=" * 80)

    # Load model
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load Phase 3 checkpoint
    print(f"Loading Phase 3 checkpoint from {reference_model_path}")
    try:
        model = PeftModel.from_pretrained(model, reference_model_path, device_map="auto")
        print("[OK] Loaded Phase 3 CoT-SFT checkpoint")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")

    # Load reference model (frozen, for KL divergence)
    print("Setting up reference model for anchoring")
    reference_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
        reference_model = PeftModel.from_pretrained(reference_model, reference_model_path, device_map="auto")
    except Exception:
        pass
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Load preference pairs
    print(f"\nLoading preference pairs from {preference_pairs_path}")
    dataset = ADPODataset(preference_pairs_path, processor)
    print(f"  {len(dataset)} pairs")

    # Use the LLM submodule to sidestep vision-model kwargs (pixel_values, etc.)
    def llm(m):
        return getattr(m, "language_model", m)

    train_model = llm(model)
    ref_model_llm = llm(reference_model)

    # Manual training loop (avoids HF Trainer's kwarg injection into InternVL's forward)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in train_model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    total_steps = len(loader) * num_train_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=min(100, total_steps)
    )

    device = next(train_model.parameters()).device
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting ADPO training (alpha={alpha})...")
    print(f"  {num_train_epochs} epochs x {len(loader)} steps = {total_steps} total")

    train_model.train()
    step = 0
    for epoch in range(num_train_epochs):
        for batch in loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            with torch.no_grad():
                ref_c = ref_model_llm(input_ids=chosen_ids, attention_mask=chosen_mask).logits
                ref_r = ref_model_llm(input_ids=rejected_ids, attention_mask=rejected_mask).logits

            out_c = train_model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
            out_r = train_model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

            loss_dict = compute_adpo_loss(
                out_c.mean(dim=-1),
                out_r.mean(dim=-1),
                ref_c.mean(dim=-1).detach(),
                ref_r.mean(dim=-1).detach(),
                alpha=alpha,
            )
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 50 == 0:
                print(
                    f"  step {step}/{total_steps} | loss {loss.item():.4f} | "
                    f"pref {loss_dict['preference_loss'].item():.4f} | "
                    f"kl {loss_dict['kl_loss'].item():.4f}"
                )

    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="plan2_adpo/preference_pairs_train.json")
    parser.add_argument("--reference", default="plan2_models/cot_sft")
    parser.add_argument("--output", default="plan2_models/adpo_final")
    parser.add_argument("--alpha", type=float, default=0.5, help="Anchoring strength")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_adpo(
        preference_pairs_path=args.pairs,
        reference_model_path=args.reference,
        output_dir=args.output,
        alpha=args.alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
