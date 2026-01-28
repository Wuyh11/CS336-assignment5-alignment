#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Problem (math_baseline) reference implementation (single-GPU friendly).

What it does (as required by the PDF):
  1) Load MATH validation examples from a jsonl file
  2) Format prompts with the r1_zero prompt
  3) Generate model outputs with vLLM
  4) Compute evaluation metrics with r1_zero_reward_fn
  5) Serialize examples + generations + scores to disk
And prints the requested category counts:
  (format=1, answer=1), (format=1, answer=0), (format=0, answer=0)

PDF reference: Section 3.2 + Problem (math_baseline) + stop at </answer>.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import torch

# vLLM
from vllm import LLM, SamplingParams

# Reward function provided by the assignment codebase
# (fast answer parser used in reasoning RL work)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


# -------------------------
# Utilities
# -------------------------

def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str | os.PathLike, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_text_file(path: str | os.PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_r1_zero_prompt_template(repo_root: Path | None = None) -> str:
    """
    Prefer reading the prompt from cs336_alignment/prompts/r1_zero.prompt
    (as the PDF mentions). If not found, fall back to an embedded template.
    """
    if repo_root is not None:
        prompt_path = repo_root / "cs336_alignment" / "prompts" / "r1_zero.prompt"
        if prompt_path.exists():
            return load_text_file(prompt_path)

    # Fallback (matches the PDF text)
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant\n"
        "solves it. The Assistant first thinks about the reasoning process in the mind and\n"
        "then provides the User with the answer. The reasoning process is enclosed within\n"
        "<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,\n"
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        "\n"
        "User: {question}\n"
        "Assistant: <think>\n"
    )


def format_prompts(examples: list[dict[str, Any]], prompt_template: str) -> list[str]:
    """
    The MATH jsonl usually has a "problem" field for the question.
    We also support "question" as a fallback.
    """
    prompts = []
    for ex in examples:
        q = ex.get("problem", None)
        if q is None:
            q = ex.get("question", None)
        if q is None:
            raise KeyError("Each example must have a 'problem' or 'question' field.")
        prompts.append(prompt_template.format(question=q))
    return prompts


def get_ground_truths(examples: list[dict[str, Any]]) -> list[str]:
    """
    The MATH jsonl usually has an "answer" field for the gold answer.
    We also support "final_answer" as a fallback.
    """
    gts = []
    for ex in examples:
        a = ex.get("answer", None)
        if a is None:
            a = ex.get("final_answer", None)
        if a is None:
            raise KeyError("Each example must have an 'answer' (or 'final_answer') field.")
        gts.append(a)
    return gts


# -------------------------
# Core evaluation
# -------------------------

@dataclass
class EvalSummary:
    n: int
    mean_reward: float
    mean_format_reward: float
    mean_answer_reward: float
    # category counts requested in (b)
    n_fmt1_ans1: int
    n_fmt1_ans0: int
    n_fmt0_ans0: int


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> tuple[list[dict[str, Any]], EvalSummary]:
    """
    Run batched generation with vLLM, score with reward_fn, and return rows + summary.

    The reward_fn (r1_zero_reward_fn) returns a dict with:
      - "reward"
      - "format_reward"
      - "answer_reward"
    """
    assert len(prompts) == len(ground_truths)
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    rows: list[dict[str, Any]] = []
    rewards = []
    fmt_rewards = []
    ans_rewards = []

    c_fmt1_ans1 = 0
    c_fmt1_ans0 = 0
    c_fmt0_ans0 = 0

    for i, out in enumerate(outputs):
        prompt = out.prompt
        # vLLM returns a list of candidates (we use the first because n=1 here)
        completion = out.outputs[0].text

        gt = ground_truths[i]
        score = reward_fn(completion, gt)

        r = float(score.get("reward", 0.0))
        fr = float(score.get("format_reward", 0.0))
        ar = float(score.get("answer_reward", 0.0))

        rewards.append(r)
        fmt_rewards.append(fr)
        ans_rewards.append(ar)

        # categories required by the writeup
        if fr == 1.0 and ar == 1.0:
            c_fmt1_ans1 += 1
        elif fr == 1.0 and ar == 0.0:
            c_fmt1_ans0 += 1
        elif fr == 0.0 and ar == 0.0:
            c_fmt0_ans0 += 1
        else:
            # Usually shouldn't happen in this assignment's reward function
            pass

        rows.append(
            {
                "idx": i,
                "prompt": prompt,
                "completion": completion,
                "ground_truth": gt,
                "scores": {
                    "reward": r,
                    "format_reward": fr,
                    "answer_reward": ar,
                },
            }
        )

    n = len(rows)
    mean_reward = sum(rewards) / max(n, 1)
    mean_fr = sum(fmt_rewards) / max(n, 1)
    mean_ar = sum(ans_rewards) / max(n, 1)

    summary = EvalSummary(
        n=n,
        mean_reward=mean_reward,
        mean_format_reward=mean_fr,
        mean_answer_reward=mean_ar,
        n_fmt1_ans1=c_fmt1_ans1,
        n_fmt1_ans0=c_fmt1_ans0,
        n_fmt0_ans0=c_fmt0_ans0,
    )
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local path (e.g. /path/to/Qwen2.5-Math-1.5B)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to MATH validation.jsonl (or any jsonl with fields problem/answer).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/math_baseline/results.jsonl",
        help="Where to write the per-example results jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only evaluate first N examples (useful for quick local sanity checks).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for vLLM sampling.",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.90,
        help="vLLM gpu_memory_utilization (A10 24GB usually ok with 0.85~0.92).",
    )
    args = parser.parse_args()

    # Load data
    examples = read_jsonl(args.data)
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    # Prompt template (read from repo if possible)
    repo_root = Path(__file__).resolve().parents[1]  # assumes scripts/ under repo
    prompt_template = get_r1_zero_prompt_template(repo_root=repo_root)
    prompts = format_prompts(examples, prompt_template)
    gts = get_ground_truths(examples)

    # vLLM sampling params (as specified in the PDF)
    # temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
    # include_stop_str_in_output=True
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    # Init vLLM (single-GPU A10)
    # Notes:
    # - dtype=bfloat16 is generally best if your stack supports it
    # - tensor_parallel_size=1 for single GPU
    llm = LLM(
        model=args.model,
        device="cuda",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
    )

    rows, summary = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=gts,
        eval_sampling_params=sampling_params,
    )

    # Save per-example rows
    write_jsonl(args.out, rows)

    # Save a tiny summary json next to it
    summary_path = str(Path(args.out).with_suffix(".summary.json"))
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)

    # Print summary
    print("=== math_baseline summary ===")
    print(f"n = {summary.n}")
    print(f"mean_reward = {summary.mean_reward:.4f}")
    print(f"mean_format_reward = {summary.mean_format_reward:.4f}")
    print(f"mean_answer_reward = {summary.mean_answer_reward:.4f}")
    print("category counts:")
    print(f"  (format=1, answer=1): {summary.n_fmt1_ans1}")
    print(f"  (format=1, answer=0): {summary.n_fmt1_ans0}")
    print(f"  (format=0, answer=0): {summary.n_fmt0_ans0}")
    print(f"saved: {args.out}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    # Make CUDA errors easier to diagnose
    torch.set_float32_matmul_precision("high")
    main()