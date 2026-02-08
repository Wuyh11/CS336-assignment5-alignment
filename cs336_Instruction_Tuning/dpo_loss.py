from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerBase


def _alpaca_prompt(instruction: str) -> str:
    """
    Minimal Alpaca-style instruction template used in many CS336 assignments.
    If your codebase defines a canonical template elsewhere, prefer importing it.
    """
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def _sequence_logprob(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute log p(x_0, x_1, ..., x_{T-1}) under an autoregressive LM, i.e.
    sum_{t=1..T-1} log p(x_t | x_<t), masked by attention_mask.

    Returns: shape (batch,), on the same device as model.
    """
    # Forward pass: logits shape (B, T, V)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # We predict token t using logits at position t-1, so shift.
    # logits[:, :-1, :] predicts input_ids[:, 1:]
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:].to(dtype=torch.bool)

    # Numerically stable log-softmax, in float32 for stability (bf16/fp16 logits can underflow).
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T-1, V)

    # Gather the log-prob assigned to the actual next token at each position.
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Mask out padding positions (or anything not attended).
    token_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)

    # Sum over time to get sequence logprob.
    return token_log_probs.sum(dim=-1)  # (B,)


def per_instance_dpo(
    policy_model: nn.Module,
    reference_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    chosen: str,
    rejected: str,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute the per-instance DPO loss (Equation 3 in the handout).

    Returns:
        loss: scalar tensor on the SAME device as policy_model.
    """
    # -----------------------------
    # 1) Decide devices
    # -----------------------------
    policy_device = next(policy_model.parameters()).device
    ref_device = next(reference_model.parameters()).device

    # -----------------------------
    # 2) Build Alpaca-formatted full texts and append EOS after each response
    # -----------------------------
    eos = tokenizer.eos_token or ""
    prefix = _alpaca_prompt(prompt)
    chosen_text = prefix + chosen + eos
    rejected_text = prefix + rejected + eos

    # -----------------------------
    # 3) Tokenize (we will move tensors to each model's device separately)
    # -----------------------------
    chosen_batch = tokenizer(chosen_text, return_tensors="pt", add_special_tokens=False)
    rejected_batch = tokenizer(rejected_text, return_tensors="pt", add_special_tokens=False)

    # -----------------------------
    # 4) Policy logprobs (need gradients)
    # -----------------------------
    chosen_ids_pol = chosen_batch["input_ids"].to(policy_device)
    chosen_mask_pol = chosen_batch["attention_mask"].to(policy_device)
    rejected_ids_pol = rejected_batch["input_ids"].to(policy_device)
    rejected_mask_pol = rejected_batch["attention_mask"].to(policy_device)

    logp_pol_chosen = _sequence_logprob(policy_model, chosen_ids_pol, chosen_mask_pol)  # (1,)
    logp_pol_rejected = _sequence_logprob(policy_model, rejected_ids_pol, rejected_mask_pol)  # (1,)
    delta_pol = logp_pol_chosen - logp_pol_rejected  # (1,)

    # -----------------------------
    # 5) Reference logprobs (no gradients, may be on another device)
    # -----------------------------
    chosen_ids_ref = chosen_batch["input_ids"].to(ref_device)
    chosen_mask_ref = chosen_batch["attention_mask"].to(ref_device)
    rejected_ids_ref = rejected_batch["input_ids"].to(ref_device)
    rejected_mask_ref = rejected_batch["attention_mask"].to(ref_device)

    with torch.no_grad():
        logp_ref_chosen = _sequence_logprob(reference_model, chosen_ids_ref, chosen_mask_ref)  # (1,)
        logp_ref_rejected = _sequence_logprob(reference_model, rejected_ids_ref, rejected_mask_ref)  # (1,)
        delta_ref = logp_ref_chosen - logp_ref_rejected  # (1,)

    # Move reference delta onto policy device so the final loss sits with policy params.
    delta_ref = delta_ref.to(policy_device)

    # -----------------------------
    # 6) DPO loss:  -log sigmoid( beta * (delta_pol - delta_ref) )
    # Use logsigmoid for numerical stability.
    # -----------------------------
    logits = beta * (delta_pol - delta_ref)  # (1,)
    loss = -F.logsigmoid(logits)             # (1,)

    # Return scalar
    return loss.mean()