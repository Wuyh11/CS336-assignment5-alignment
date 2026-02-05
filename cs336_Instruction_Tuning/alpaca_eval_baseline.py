from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from vllm import LLM, SamplingParams


# 2.3 里说：prompting setup 只需要 {instruction} 本身 :contentReference[oaicite:1]{index=1}
# 同时 2.0 Motivation 里说：所有任务统一使用同一个 zero-shot system prompt，
# 并且看到 "# Query:" 就 stop。:contentReference[oaicite:2]{index=2}
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


def load_system_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_SYSTEM_PROMPT


def format_prompt(system_prompt: str, instruction: str) -> str:
    # system prompt 里用 {instruction} 占位
    return system_prompt.format(instruction=instruction)


def load_alpaca_eval_dataset(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """
    AlpacaEval 数据在不同版本里可能是：
      - JSON 数组文件：[{...}, {...}]
      - 或 JSONL：每行一个 {...}

    我们尽量兼容两者，只要每条样例里包含：
      - instruction: str
      - dataset: str   (作业要求输出里要保留该字段) :contentReference[oaicite:3]{index=3}
    """
    text = path.read_text(encoding="utf-8").strip()
    examples: List[Dict[str, Any]] = []

    if not text:
        return examples

    # 1) 先尝试按 JSON 数组读
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file is not a list")
        for ex in data:
            examples.append(ex)
            if limit is not None and len(examples) >= limit:
                break
        return examples

    # 2) 否则按 JSONL 读
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(ex)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="/data/a5-alignment/models/Llama-3.1-8B",
        help="Together 集群上 Llama 3.1 8B Base 的路径（作业建议用该路径避免重复下载）。",
    )
    ap.add_argument(
        "--alpaca_eval_path",
        type=str,
        required=True,
        help="AlpacaEval 数据路径（在仓库 data/* 下，按你实际文件名传入）。",
    )
    ap.add_argument(
        "--system_prompt_path",
        type=str,
        default="cs336_Instruction_Tuning/prompts/zero_shot_system_prompt.txt",
        help="zero-shot system prompt 文件路径（仓库一般提供）。",
    )
    ap.add_argument(
        "--out_path",
        type=str,
        default="outputs/alpaca_eval_zero_shot_predictions.json",
        help="输出 JSON 数组文件路径（必须是 JSON array，不能是 jsonl）。",
    )
    ap.add_argument("--generator", type=str, default="llama-3.1-8b-base", help="写入每条结果的 generator 字段。")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 表示不截断；否则只生成前 N 条。")
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    args = ap.parse_args()

    dataset_path = Path(args.alpaca_eval_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt(Path(args.system_prompt_path))

    limit = None if args.limit == 0 else args.limit
    eval_set = load_alpaca_eval_dataset(dataset_path, limit=limit)

    # 基本校验：确保 instruction/dataset 存在（作业明确要求输出里要有这些键）:contentReference[oaicite:4]{index=4}
    for i, ex in enumerate(eval_set):
        if "instruction" not in ex:
            raise KeyError(f"Example {i} missing key 'instruction'")
        if "dataset" not in ex:
            raise KeyError(f"Example {i} missing key 'dataset'")

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem_util)

    # 2.3 指定 greedy decoding：temperature=0.0, top_p=1.0 :contentReference[oaicite:5]{index=5}
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        # Motivation 里说明：看到 "# Query:" 就 stop（system prompt 约定）:contentReference[oaicite:6]{index=6}
        stop=["# Query:"],
    )

    prompts: List[str] = []
    for ex in eval_set:
        instruction = ex["instruction"]
        # 2.3 的“task prompt”就是 instruction；但我们仍嵌进统一 system prompt :contentReference[oaicite:7]{index=7}
        prompts.append(format_prompt(system_prompt, instruction))

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    t1 = time.time()

    # 组装作业要求的 JSON array：
    # 每条必须含 instruction/output/generator/dataset :contentReference[oaicite:8]{index=8}
    result_list: List[Dict[str, Any]] = []
    for ex, out in zip(eval_set, outputs):
        gen_text = out.outputs[0].text

        result_list.append(
            {
                "instruction": ex["instruction"],
                "output": gen_text,
                "generator": args.generator,
                "dataset": ex["dataset"],
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    total = len(result_list)
    dt = max(t1 - t0, 1e-9)
    print("========== AlpacaEval zero-shot output collection ==========")
    print(f"Saved -> {out_path}")
    print(f"Examples: {total}")
    print(f"Total generation time: {dt:.2f}s")
    print(f"Throughput: {total / dt:.2f} examples/s")


if __name__ == "__main__":
    main()