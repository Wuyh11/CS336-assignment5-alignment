#!/usr/bin/env python3
# cs336_alignment/scripts/eval_mmlu_zero_shot.py
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vllm import LLM, SamplingParams

from cs336_Instruction_Tuning.mmlu_baseline import parse_mmlu_response

# -------------------------
# Prompt templates (来自 PDF 2.1)
# -------------------------
MMLU_TASK_PROMPT = """Answer the following multiple choice question about {subject}. Respond with a single
sentence of the form "The correct answer is _", filling the blank with the letter
corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer:"""

# system prompt（建议直接读 cs336_alignment/prompts/zero_shot_system_prompt.txt）
# 这里提供一个内置 fallback，避免你没读文件脚本直接报错。
DEFAULT_SYSTEM_PROMPT = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.
# Query:
```{instruction}```
# Answer:
```"""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_full_prompt(system_prompt: str, instruction: str) -> str:
    """
    system_prompt 里包含 ```{instruction}``` 的占位符；
    我们直接 .format(instruction=...) 即可。
    """
    return system_prompt.format(instruction=instruction)


def normalize_gold_answer(ans: Any) -> str:
    """
    兼容常见标注格式：
      - "A"/"B"/"C"/"D"
      - 0/1/2/3
      - "0"/"1"/"2"/"3"
    """
    if isinstance(ans, str):
        a = ans.strip()
        if a.upper() in {"A", "B", "C", "D"}:
            return a.upper()
        if a.isdigit():
            idx = int(a)
            return "ABCD"[idx]
    if isinstance(ans, int):
        return "ABCD"[ans]
    raise ValueError(f"Unrecognized answer format: {ans!r}")


def load_mmlu_examples(dataset_path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """
    读取 MMLU jsonl。字段尽量兼容：
      subject: str
      question: str
      options/choices: list[str] 长度 4
      answer: "A"/... 或 0..3
    """
    examples: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            subject = ex.get("subject") or ex.get("task") or ex.get("category") or "unknown"
            question = ex.get("question") or ex.get("query") or ex.get("prompt")

            options = ex.get("options") or ex.get("choices") or ex.get("answer_choices")
            if options is None:
                raise KeyError(f"Cannot find options/choices in example keys: {list(ex.keys())}")

            if len(options) != 4:
                raise ValueError(f"Expected 4 options, got {len(options)}")

            answer = ex.get("answer") or ex.get("label") or ex.get("gold")
            if answer is None:
                raise KeyError(f"Cannot find answer/label/gold in example keys: {list(ex.keys())}")

            examples.append({
                "subject": subject,
                "question": question,
                "options": options,
                "gold": normalize_gold_answer(answer),
                "raw": ex,
            })
            if limit is not None and len(examples) >= limit:
                break
    return examples


def format_mmlu_instruction(ex: Dict[str, Any]) -> str:
    a, b, c, d = ex["options"]
    return MMLU_TASK_PROMPT.format(
        subject=ex["subject"],
        question=ex["question"],
        a=a, b=b, c=c, d=d,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/data/a5-alignment/models/Llama-3.1-8B",
                    help="Llama 3.1 8B base path (Together cluster path).")
    ap.add_argument("--mmlu_path", type=str, required=True,
                    help="Path to MMLU jsonl (e.g., data/mmlu/test.jsonl).")
    ap.add_argument("--system_prompt_path", type=str, default="cs336_alignment/prompts/zero_shot_system_prompt.txt",
                    help="Path to zero-shot system prompt text file.")
    ap.add_argument("--out_path", type=str, default="outputs/mmlu_zero_shot_results.jsonl",
                    help="Where to write per-example results (jsonl).")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise evaluate only first N.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    args = ap.parse_args()

    random.seed(args.seed)

    mmlu_path = Path(args.mmlu_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读 system prompt（有文件就读文件，否则用 fallback）
    sp_path = Path(args.system_prompt_path)
    if sp_path.exists():
        system_prompt = load_text(sp_path)
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    limit = None if args.limit == 0 else args.limit
    examples = load_mmlu_examples(mmlu_path, limit=limit)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_util)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["# Query:"],  # PDF 里建议看到 # Query: 就停止
    )

    prompts: List[str] = []
    for ex in examples:
        instruction = format_mmlu_instruction(ex)
        prompts.append(build_full_prompt(system_prompt, instruction))

    # 计时：吞吐 (d)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t1 = time.time()

    total = len(outputs)
    total_time = max(t1 - t0, 1e-9)
    throughput = total / total_time

    # 打分：accuracy + parse fail (c)(e)
    correct = 0
    parse_fail = 0
    results: List[Dict[str, Any]] = []

    for ex, out in zip(examples, outputs):
        text = out.outputs[0].text
        pred = parse_mmlu_response(text)
        gold = ex["gold"]
        ok_parse = pred is not None
        if not ok_parse:
            parse_fail += 1
        is_correct = (pred == gold)
        if is_correct:
            correct += 1

        results.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "options": ex["options"],
            "gold": gold,
            "pred": pred,
            "parsed": ok_parse,
            "correct": is_correct,
            "generation": text,
            "raw_example": ex["raw"],
        })

    acc = correct / max(total, 1)

    # 序列化 (b)(5)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # (f) 抽 10 个错误例子做粗分析（脚本里打印出来；你也可打开 out_path 看）
    wrong = [r for r in results if (r["pred"] is None) or (r["pred"] != r["gold"])]
    random.shuffle(wrong)
    sample_wrong = wrong[:10]

    print("========== MMLU Zero-shot Report ==========")
    print(f"Examples: {total}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Parse fail: {parse_fail} ({parse_fail/ max(total,1):.2%})")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"Saved results -> {out_path}")
    print("\n========== Sample 10 Wrong Examples ==========")
    for i, r in enumerate(sample_wrong, 1):
        print(f"\n--- Wrong #{i} ---")
        print(f"Subject: {r['subject']}")
        print(f"Gold: {r['gold']} | Pred: {r['pred']} | Parsed: {r['parsed']}")
        print(f"Question: {r['question']}")
        print("Model generation:")
        print(r["generation"])


if __name__ == "__main__":
    main()