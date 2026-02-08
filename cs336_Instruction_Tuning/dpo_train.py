#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dpo_train.py — Reference implementation for CS336 A5 supplement (Section 5.4 DPO Training).

What this script does (aligned with the handout 5.4):
  1) Load Anthropic HH preference data (chosen/rejected) and convert each example to:
     prompt (instruction) + chosen response + rejected response (single-turn only).
  2) Load TWO copies of the SFT model:
       - policy_model (trainable) on GPU A
       - reference_model (frozen) on GPU B
  3) Train policy_model for 1 epoch using per-instance DPO loss with gradient accumulation
     (effective batch size = grad_accum_steps, since microbatch=1).
  4) Track training loss and validation "classification accuracy":
       correct if logp(chosen|prompt) > logp(rejected|prompt).
  5) Save the model checkpoint with the highest validation accuracy.

No Trainer. No RL sampling. Just log-probabilities, per the DPO observation in 5.3/5.4.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


# -----------------------------
# 0) Reproducibility helpers
# -----------------------------
def set_seed(seed: int) -> None:
    """Set python + torch seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# 1) Alpaca template (same as SFT)
# -----------------------------
def alpaca_prefix(instruction: str) -> str:
    """
    Alpaca prompt template used in the handout for SFT.
    We reuse it for DPO as required in 5.3.
    """
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


# -----------------------------
# 2) HH data loading (single-turn only)
# -----------------------------
@dataclass
class HHExample:
    """A single preference pair for DPO."""
    instruction: str
    chosen: str
    rejected: str
    source_file: str  # which HH split file this came from (for analysis/debug)


def _extract_first_human_and_assistant(conversation_text: str) -> Tuple[str, str] | None:
    """
    HH 'chosen'/'rejected' fields contain a text transcript with markers like:
      'Human: ...\n\nAssistant: ...'
    We keep ONLY single-turn conversations:
      - exactly one 'Human:' message
      - take the first 'Assistant:' reply after it

    Returns (human_msg, assistant_msg) or None if multi-turn / malformed.
    """
    # Split by the marker "Human:"; the first chunk before it is usually empty/preamble.
    parts = conversation_text.split("Human:")
    if len(parts) < 2:
        return None

    # If there is more than one human message, it's multi-turn (ignore per spec).
    # A strict way: count occurrences of "\n\nHuman:" or "Human:" markers.
    # Here we approximate: if "Human:" appears more than once, skip.
    if conversation_text.count("Human:") != 1:
        return None

    # After "Human:" we expect "... Assistant: ..."
    after_human = parts[1]
    if "Assistant:" not in after_human:
        return None

    human_msg, after_assistant = after_human.split("Assistant:", 1)

    human_msg = human_msg.strip()
    assistant_msg = after_assistant.strip()

    # If assistant contains another "Human:" marker, it became multi-turn; skip.
    if "Human:" in assistant_msg:
        return None

    if not human_msg or not assistant_msg:
        return None

    return human_msg, assistant_msg


def load_hh_dataset(hh_dir: str) -> List[HHExample]:
    """
    Load and combine the 4 HH jsonl.gz training files as described in the handout.
    Each line is a JSON object with keys 'chosen' and 'rejected'.

    Processing steps (per handout):
      - ignore multi-turn (human >1 message)
      - split into instruction + chosen response + rejected response
      - keep track of which file it came from
    """
    hh_path = Path(hh_dir)
    files = [
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz",
    ]

    examples: List[HHExample] = []
    for fname in files:
        fpath = hh_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing HH file: {fpath}")

        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                chosen_text = obj.get("chosen", "")
                rejected_text = obj.get("rejected", "")

                chosen_pair = _extract_first_human_and_assistant(chosen_text)
                rejected_pair = _extract_first_human_and_assistant(rejected_text)
                if chosen_pair is None or rejected_pair is None:
                    continue

                instr_c, chosen_ans = chosen_pair
                instr_r, rejected_ans = rejected_pair

                # We only keep examples where the instruction is the same for chosen and rejected.
                if instr_c.strip() != instr_r.strip():
                    continue

                examples.append(
                    HHExample(
                        instruction=instr_c.strip(),
                        chosen=chosen_ans.strip(),
                        rejected=rejected_ans.strip(),
                        source_file=fname,
                    )
                )
    return examples


# -----------------------------
# 3) Log-probability utilities
# -----------------------------
def sequence_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log p(x_0..x_{T-1}) for an autoregressive LM:
      sum_{t=1..T-1} log p(x_t | x_<t)
    Masked by attention_mask.

    Returns shape (B,) on model's device.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)

    # Shift: logits at position t-1 predict token t.
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].to(dtype=torch.bool)

    # Use float32 log_softmax for numerical stability (bf16/fp16 can underflow).
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T-1, V)

    # Gather log-prob of the actual next tokens.
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Ignore padding tokens.
    token_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)

    return token_log_probs.sum(dim=-1)  # (B,)


def per_instance_dpo_loss(
    policy_model: nn.Module,
    reference_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    chosen: str,
    rejected: str,
    beta: float,
) -> torch.Tensor:
    """
    Per-instance DPO loss (Eq. 3), implemented via the 'prompt cancels out' observation:
      Δ = log π(x⊕yw) - log π(x⊕yl)
    so we don't need to separately compute log π(prompt).

    Returns:
      scalar loss on the policy device (important if ref is on another GPU).
    """
    policy_device = next(policy_model.parameters()).device
    ref_device = next(reference_model.parameters()).device

    # Build Alpaca-formatted texts and append EOS after response (handout requirement).
    eos = tokenizer.eos_token or ""
    prefix = alpaca_prefix(instruction)
    chosen_text = prefix + chosen + eos
    rejected_text = prefix + rejected + eos

    # Tokenize once on CPU; we will move tensors to each device as needed.
    chosen_batch = tokenizer(chosen_text, return_tensors="pt", add_special_tokens=False)
    rejected_batch = tokenizer(rejected_text, return_tensors="pt", add_special_tokens=False)

    # Policy forward (requires grads).
    chosen_ids_pol = chosen_batch["input_ids"].to(policy_device)
    chosen_mask_pol = chosen_batch["attention_mask"].to(policy_device)
    rejected_ids_pol = rejected_batch["input_ids"].to(policy_device)
    rejected_mask_pol = rejected_batch["attention_mask"].to(policy_device)

    logp_pol_chosen = sequence_logprob(policy_model, chosen_ids_pol, chosen_mask_pol)      # (1,)
    logp_pol_rejected = sequence_logprob(policy_model, rejected_ids_pol, rejected_mask_pol)  # (1,)
    delta_pol = logp_pol_chosen - logp_pol_rejected  # (1,)

    # Reference forward (no grads, may be on another GPU).
    chosen_ids_ref = chosen_batch["input_ids"].to(ref_device)
    chosen_mask_ref = chosen_batch["attention_mask"].to(ref_device)
    rejected_ids_ref = rejected_batch["input_ids"].to(ref_device)
    rejected_mask_ref = rejected_batch["attention_mask"].to(ref_device)

    with torch.no_grad():
        logp_ref_chosen = sequence_logprob(reference_model, chosen_ids_ref, chosen_mask_ref)        # (1,)
        logp_ref_rejected = sequence_logprob(reference_model, rejected_ids_ref, rejected_mask_ref)  # (1,)
        delta_ref = logp_ref_chosen - logp_ref_rejected  # (1,)

    # Move ref result onto policy device so loss is on the correct device for backward().
    delta_ref = delta_ref.to(policy_device)

    # DPO: -log sigmoid(beta * (delta_pol - delta_ref)) ; use logsigmoid for stability.
    logits = beta * (delta_pol - delta_ref)  # (1,)
    loss = -F.logsigmoid(logits)             # (1,)

    return loss.mean()  # scalar


@torch.no_grad()
def per_instance_pref_correct(
    policy_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    chosen: str,
    rejected: str,
) -> bool:
    """
    Validation 'classification accuracy' in 5.4:
      correct if logp(chosen | prompt) > logp(rejected | prompt).
    Implemented via unconditional diff logp(x⊕chosen) - logp(x⊕rejected).
    """
    device = next(policy_model.parameters()).device

    eos = tokenizer.eos_token or ""
    prefix = alpaca_prefix(instruction)
    chosen_text = prefix + chosen + eos
    rejected_text = prefix + rejected + eos

    chosen_batch = tokenizer(chosen_text, return_tensors="pt", add_special_tokens=False)
    rejected_batch = tokenizer(rejected_text, return_tensors="pt", add_special_tokens=False)

    chosen_ids = chosen_batch["input_ids"].to(device)
    chosen_mask = chosen_batch["attention_mask"].to(device)
    rejected_ids = rejected_batch["input_ids"].to(device)
    rejected_mask = rejected_batch["attention_mask"].to(device)

    logp_chosen = sequence_logprob(policy_model, chosen_ids, chosen_mask)      # (1,)
    logp_rejected = sequence_logprob(policy_model, rejected_ids, rejected_mask)  # (1,)

    return bool((logp_chosen > logp_rejected).item())


# -----------------------------
# 4) Training loop
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # Paths.
    parser.add_argument("--sft_model_path", type=str, required=True,
                        help="Path to the SFT model directory (used to init both policy and reference).")
    parser.add_argument("--hh_dir", type=str, required=True,
                        help="Directory containing HH jsonl.gz files (4 train splits).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save best DPO checkpoint.")

    # Hyperparameters (handout suggests: batch=64, beta=0.1, lr=1e-6).
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--grad_accum_steps", type=int, default=64,
                        help="Effective batch size when microbatch=1.")
    parser.add_argument("--val_size", type=int, default=200)

    # Logging / eval.
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=200)

    # Misc.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_examples", type=int, default=-1,
                        help="For debugging: limit number of train examples. -1 means full dataset.")

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Device placement (2 GPUs recommended)
    # -----------------------------
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("This script requires at least 1 CUDA GPU.")
    if n_gpus == 1:
        # Still runnable for debugging, but memory will likely be tight.
        policy_device = torch.device("cuda:0")
        ref_device = torch.device("cuda:0")
        print("[WARN] Only 1 GPU detected. policy/ref will share the same device; may OOM.")
    else:
        policy_device = torch.device("cuda:0")
        ref_device = torch.device("cuda:1")

    # -----------------------------
    # Load tokenizer (shared)
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    # Make sure tokenizer has a pad_token for attention_mask; if missing, reuse eos as pad.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------
    # Load models: two copies of SFT model
    # -----------------------------
    # Reference model: frozen, eval mode.
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,             # memory saver (handout uses bf16)
        attn_implementation="flash_attention_2" # memory/time saver if available
    ).to(ref_device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad_(False)

    # Policy model: trainable, train mode.
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(policy_device)
    policy_model.train()

    # -----------------------------
    # Optimizer: RMSprop (handout says AdamW likely too memory-heavy)
    # -----------------------------
    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=args.lr)

    # -----------------------------
    # Load HH data and split train/val
    # -----------------------------
    all_examples = load_hh_dataset(args.hh_dir)
    random.shuffle(all_examples)

    val_examples = all_examples[: args.val_size]
    train_examples = all_examples[args.val_size :]

    if args.max_train_examples > 0:
        train_examples = train_examples[: args.max_train_examples]

    print(f"Loaded HH examples: total={len(all_examples)} train={len(train_examples)} val={len(val_examples)}")

    # -----------------------------
    # Training
    # -----------------------------
    best_val_acc = -1.0
    global_step = 0

    # For plotting curves later, we store (step, loss, val_acc) in a jsonl log.
    metrics_path = out_dir / "dpo_metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as mf:
        pass  # create/truncate file

    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory vs filling zeros

    for i, ex in enumerate(train_examples):
        # 1) Compute per-instance DPO loss.
        loss = per_instance_dpo_loss(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            instruction=ex.instruction,
            chosen=ex.chosen,
            rejected=ex.rejected,
            beta=args.beta,
        )

        # 2) Gradient accumulation: divide loss so summed grads become averaged grads.
        loss = loss / args.grad_accum_steps
        loss.backward()

        # 3) Step optimizer every grad_accum_steps microbatches.
        if (i + 1) % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # 4) Logging training loss (multiply back by grad_accum_steps for human-readable scale).
            if global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[step={global_step:06d}] "
                    f"train_loss={loss.item() * args.grad_accum_steps:.4f} "
                    f"elapsed={elapsed/60:.1f}m"
                )

            # 5) Periodic validation accuracy (classification accuracy).
            if global_step % args.eval_every == 0:
                policy_model.eval()
                correct = 0
                for vx in val_examples:
                    if per_instance_pref_correct(policy_model, tokenizer, vx.instruction, vx.chosen, vx.rejected):
                        correct += 1
                val_acc = correct / max(1, len(val_examples))
                policy_model.train()

                # Write metrics record for plotting.
                record = {
                    "step": global_step,
                    "train_loss": float(loss.item() * args.grad_accum_steps),
                    "val_acc": float(val_acc),
                    "beta": args.beta,
                    "lr": args.lr,
                    "grad_accum_steps": args.grad_accum_steps,
                }
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(record) + "\n")

                print(f"[step={global_step:06d}] val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

                # Save best checkpoint.
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt_dir = out_dir / "best_ckpt"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    policy_model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"  -> Saved new best checkpoint to: {ckpt_dir}")

    # If the number of train examples isn't divisible by grad_accum_steps,
    # we may have gradients pending; optionally apply one last optimizer step.
    # This keeps behavior deterministic and avoids dropping the tail.
    if len(train_examples) % args.grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        print(f"[final] Applied last optimizer step for leftover microbatches. step={global_step}")

    print(f"Done. Best val_acc={best_val_acc:.4f}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Best checkpoint in: {out_dir / 'best_ckpt'}")


if __name__ == "__main__":
    main()