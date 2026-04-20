#!/usr/bin/env python3
"""
Phase 5: ADPO (Anchored Direct Preference Optimization) Training.

Trains with ADPO loss on preference pairs:
  L_ADPO = L_preference(chosen > rejected) + alpha * L_anchor(model | reference)

Where:
  - L_preference: Bradley-Terry ranking loss
  - L_anchor: KL divergence vs. LoRA-disabled (reference) forward
  - alpha: anchoring strength (swept 0.1-1.0 to find optimal)
"""

import json
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


class ADPODataset(Dataset):
    def __init__(self, pairs_path: str, tokenizer, max_length: int = 1024):
        with open(pairs_path) as f:
            raw = json.load(f)
        # Filter to pairs with at least one rejected
        self.pairs = [p for p in raw if p.get("rejected")]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def _tok(self, text):
        out = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = f"Question: {pair['prompt']}\n\nAnswer:"
        chosen_ans = pair["chosen"]["answer"]
        if pair["chosen"].get("reasoning"):
            chosen_text = (
                f"{prompt}\nReasoning:\n{pair['chosen']['reasoning']}\n"
                f"Answer: {chosen_ans}"
            )
        else:
            chosen_text = f"{prompt} {chosen_ans}"

        rejected_ans = pair["rejected"][0]["answer"]
        rejected_text = f"{prompt} {rejected_ans}"

        c_ids, c_mask = self._tok(chosen_text)
        r_ids, r_mask = self._tok(rejected_text)
        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
        }


def _sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence summed log-prob of input_ids under logits (shift-by-one)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = mask[:, 1:].to(shift_logits.dtype)
    logp = F.log_softmax(shift_logits.float(), dim=-1)
    token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * shift_mask).sum(dim=-1)


def compute_adpo_loss(
    pol_logp_c, pol_logp_r, ref_logp_c, ref_logp_r, alpha: float = 0.5
) -> Dict[str, torch.Tensor]:
    # DPO-style preference term: sigma((pol_c - ref_c) - (pol_r - ref_r))
    margin = (pol_logp_c - ref_logp_c) - (pol_logp_r - ref_logp_r)
    pref_loss = -F.logsigmoid(margin).mean()
    # Anchor: penalize divergence of policy from reference on chosen+rejected
    anchor = ((pol_logp_c - ref_logp_c) ** 2 + (pol_logp_r - ref_logp_r) ** 2).mean()
    loss = pref_loss + alpha * anchor
    return {"loss": loss, "preference_loss": pref_loss, "kl_loss": anchor}


def _get_llm(peft_model):
    """Navigate PeftModel -> base InternVL -> language_model submodule."""
    base = getattr(peft_model, "base_model", peft_model)
    inner = getattr(base, "model", base)
    return getattr(inner, "language_model", inner)


def train_adpo(
    preference_pairs_path: str = "plan2_adpo/preference_pairs_train.json",
    reference_model_path: str = "plan2_models/cot_sft",
    output_dir: str = "plan2_models/adpo_final",
    model_name: str = "OpenGVLab/InternVL2_5-8B",
    alpha: float = 0.5,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 1e-4,
    seed: int = 42,
    max_length: int = 1024,
):
    transformers.set_seed(seed)
    torch.manual_seed(seed)

    print("=" * 80)
    print(f"PHASE 5: ADPO TRAINING (alpha={alpha})")
    print("=" * 80)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {model_name}")
    base = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading Phase 3 adapter from {reference_model_path} (is_trainable=True)")
    model = PeftModel.from_pretrained(
        base, reference_model_path, device_map="auto", is_trainable=True
    )
    print("[OK] Loaded Phase 3 CoT-SFT adapter as trainable")

    # Enable gradient checkpointing to cut activation memory ~4x.
    # LoRA freezes base params, so we must also hook inputs to require grad
    # or GC will have no path to backprop through.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner_llm = _get_llm(model)
    for mod in (model, getattr(model, "base_model", model), inner_llm):
        if hasattr(mod, "gradient_checkpointing_enable"):
            try:
                mod.gradient_checkpointing_enable()
                print(f"[OK] gradient_checkpointing_enable on {type(mod).__name__}")
                break
            except Exception as e:
                print(f"  skip gc on {type(mod).__name__}: {e}")
    if hasattr(inner_llm, "config"):
        inner_llm.config.use_cache = False

    llm_forward = inner_llm

    print(f"\nLoading preference pairs from {preference_pairs_path}")
    dataset = ADPODataset(preference_pairs_path, tokenizer, max_length=max_length)
    print(f"  {len(dataset)} pairs (after filtering empty rejected)")

    loader = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_trainable:,}")
    if n_trainable == 0:
        raise RuntimeError("No trainable params — PeftModel is_trainable flag may be wrong")

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    total_steps = len(loader) * num_train_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=min(100, total_steps)
    )

    device = next(model.parameters()).device
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting ADPO training (alpha={alpha})")
    print(f"  {num_train_epochs} epochs x {len(loader)} steps = {total_steps} total")

    model.train()
    step = 0
    for epoch in range(num_train_epochs):
        for batch in loader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # Reference forward: disable LoRA adapter on same model
            with torch.no_grad():
                with model.disable_adapter():
                    ref_c_logits = llm_forward(input_ids=chosen_ids, attention_mask=chosen_mask).logits
                    ref_r_logits = llm_forward(input_ids=rejected_ids, attention_mask=rejected_mask).logits
                ref_lp_c = _sequence_logprob(ref_c_logits, chosen_ids, chosen_mask)
                ref_lp_r = _sequence_logprob(ref_r_logits, rejected_ids, rejected_mask)

            # Policy forward (adapter enabled, gradients on)
            pol_c_logits = llm_forward(input_ids=chosen_ids, attention_mask=chosen_mask).logits
            pol_r_logits = llm_forward(input_ids=rejected_ids, attention_mask=rejected_mask).logits
            pol_lp_c = _sequence_logprob(pol_c_logits, chosen_ids, chosen_mask)
            pol_lp_r = _sequence_logprob(pol_r_logits, rejected_ids, rejected_mask)

            loss_dict = compute_adpo_loss(pol_lp_c, pol_lp_r, ref_lp_c, ref_lp_r, alpha=alpha)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 25 == 0:
                print(
                    f"  epoch {epoch} step {step}/{total_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"pref {loss_dict['preference_loss'].item():.4f} | "
                    f"anchor {loss_dict['kl_loss'].item():.4f}",
                    flush=True,
                )

    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="plan2_adpo/preference_pairs_train.json")
    parser.add_argument("--reference", default="plan2_models/cot_sft")
    parser.add_argument("--output", default="plan2_models/adpo_final")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    train_adpo(
        preference_pairs_path=args.pairs,
        reference_model_path=args.reference,
        output_dir=args.output,
        alpha=args.alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
