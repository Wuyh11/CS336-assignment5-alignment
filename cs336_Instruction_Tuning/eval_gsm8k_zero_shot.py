from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

from cs336_Instruction_Tuning.gsm8k_baseline import parse_gsm8k_response

# GSM8K prompt（来自 PDF 2.2）
GSM8K_TASK_PROMPT = "{question}\nAnswer:"

# system prompt：建议读取 cs336_alignment/prompts/zero_shot_system_prompt.txt
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


def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def build_full_prompt(system_prompt: str, instruction: str) -> str:
    return system_prompt.format(instruction=instruction)


def parse_gold_answer(gold: Any) -> Optional[str]:
    """
    GSM8K 的 gold 有多种常见格式：
      - "72"
      - "She sold 72 clips."（某些处理版本）
      - "#### 72"（经典 GSM8K 格式）
    我们复用 parse_gsm8k_response：取最后一个数字当 gold。
    """
    if gold is None:
        return None
    if isinstance(gold, (int, float)):
        # 直接转成字符串
        return str(gold)
    if isinstance(gold, str):
        return parse_gsm8k_response(gold)
    return parse_gsm8k_response(str(gold))


def load_gsm8k_examples(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """
    读取 GSM8K jsonl，尽量兼容字段名：
      - question: str
      - answer: str / 或 "final_answer" / "label"
    """
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            question = ex.get("question") or ex.get("query") or ex.get("prompt")
            if question is None:
                raise KeyError(f"Cannot find question field in keys: {list(ex.keys())}")

            gold_raw = ex.get("answer")
            if gold_raw is None:
                gold_raw = ex.get("final_answer") or ex.get("label") or ex.get("gold")

            gold = parse_gold_answer(gold_raw)
            if gold is None:
                # gold 本身都解析不出数字，说明数据有问题；这里直接报错更好
                raise ValueError(f"Cannot parse gold answer from: {gold_raw!r}")

            examples.append({
                "question": question,
                "gold": gold,
                "raw": ex,
            })
            if limit is not None and len(examples) >= limit:
                break
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="/data/a5-alignment/models/Llama-3.1-8B",
                    help="Llama 3.1 8B base path (Together cluster path).")
    ap.add_argument("--gsm8k_path", type=str, required=True,
                    help="Path to GSM8K jsonl (e.g., data/gsm8k/test.jsonl).")
    ap.add_argument("--system_prompt_path", type=str, default="cs336_alignment/prompts/zero_shot_system_prompt.txt")
    ap.add_argument("--out_path", type=str, default="outputs/gsm8k_zero_shot_results.jsonl")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise evaluate only first N.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    args = ap.parse_args()

    random.seed(args.seed)

    gsm8k_path = Path(args.gsm8k_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sp_path = Path(args.system_prompt_path)
    system_prompt = load_text(sp_path) if sp_path.exists() else DEFAULT_SYSTEM_PROMPT

    limit = None if args.limit == 0 else args.limit
    examples = load_gsm8k_examples(gsm8k_path, limit=limit)

    # vLLM + greedy decoding（PDF 指定 temperature=0, top_p=1）
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_util)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["# Query:"],  # 与 system prompt 的 stop 约定一致
    )

    prompts: List[str] = []
    for ex in examples:
        instruction = GSM8K_TASK_PROMPT.format(question=ex["question"])
        prompts.append(build_full_prompt(system_prompt, instruction))

    # 计时 → (d) throughput
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t1 = time.time()

    total = len(outputs)
    total_time = max(t1 - t0, 1e-9)
    throughput = total / total_time

    # 打分 → (c)(e)
    correct = 0
    parse_fail = 0
    results: List[Dict[str, Any]] = []

    for ex, out in zip(examples, outputs):
        gen_text = out.outputs[0].text
        pred = parse_gsm8k_response(gen_text)
        gold = ex["gold"]

        parsed = pred is not None
        if not parsed:
            parse_fail += 1

        is_correct = (pred == gold)
        if is_correct:
            correct += 1

        results.append({
            "question": ex["question"],
            "gold": gold,
            "pred": pred,
            "parsed": parsed,
            "correct": is_correct,
            "generation": gen_text,
            "raw_example": ex["raw"],
        })

    acc = correct / max(total, 1)

    # (b) 序列化到磁盘（jsonl）
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # (f) 抽 10 个错例
    wrong = [r for r in results if (r["pred"] is None) or (r["pred"] != r["gold"])]
    random.shuffle(wrong)
    sample_wrong = wrong[:10]

    print("========== GSM8K Zero-shot Report ==========")
    print(f"Examples: {total}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Parse fail: {parse_fail} ({parse_fail / max(total,1):.2%})")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"Saved results -> {out_path}")

    print("\n========== Sample 10 Wrong Examples ==========")
    for i, r in enumerate(sample_wrong, 1):
        print(f"\n--- Wrong #{i} ---")
        print(f"Gold: {r['gold']} | Pred: {r['pred']} | Parsed: {r['parsed']}")
        print(f"Question: {r['question']}")
        print("Model generation:")
        print(r["generation"])


if __name__ == "__main__":
    main()