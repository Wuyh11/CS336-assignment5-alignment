import os
import json
import gzip
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n"
    "### Instruction:\n"
    "{prompt}\n"
    "### Response:\n"
    "{response}"
)


def _open_jsonl(path: str | os.PathLike):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _get_prompt_response(obj: Dict[str, Any]) -> tuple[str, str]:
    """
    兼容字段命名差异（不同数据集可能叫法不同）
    """
    if "prompt" in obj and "response" in obj:
        return obj["prompt"], obj["response"]
    if "instruction" in obj and "output" in obj:
        return obj["instruction"], obj["output"]
    if "input" in obj and "output" in obj:
        return obj["input"], obj["output"]
    raise KeyError(f"Unknown schema keys: {list(obj.keys())}")


class PackedSFTDataset(Dataset):
    """
    返回 dict:
      input_ids: (seq_length,)
      labels:    (seq_length,)  # next token
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
        seed: int = 0,
    ):
        super().__init__()
        assert seq_length > 1, "seq_length must be > 1"
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            # Llama 一般都有 eos，但测试环境可能 tokenizer 设置不全
            # 这里直接报错比 silent wrong 更好
            raise ValueError("tokenizer.eos_token_id is None; cannot pack documents with delimiter.")

        # 1) 读取所有 doc 文本
        docs: List[str] = []
        with _open_jsonl(dataset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt, response = _get_prompt_response(obj)
                docs.append(ALPACA_TEMPLATE.format(prompt=prompt, response=response))

        # 2) 可选：文档级 shuffle（注意是“doc 顺序”shuffle，不是 token 级）
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(docs)

        # 3) tokenize 每个 doc，并在 doc 尾部追加 eos delimiter
        all_ids = []
        for doc in docs:
            ids = tokenizer.encode(doc, add_special_tokens=False)
            # ✅ 关键：测试期望每个 doc 开头有 BOS（Llama3 的 128000）
            if bos_id is not None:
                ids = [bos_id] + ids
            # ✅ 关键：doc 之间需要 delimiter（用 eos）
            ids.append(eos_id)


            all_ids.extend(ids)

        self.all_tokens = torch.tensor(all_ids, dtype=torch.long)

        # 4) 计算可切出的 chunk 数
        # labels 要用到 s+m 位置的 token，所以总可用长度是 len(all_tokens)-1
        usable = self.all_tokens.numel() - 1
        self.n_seq = usable // self.seq_length

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if i < 0 or i >= self.n_seq:
            raise IndexError(i)
        s = i * self.seq_length
        m = self.seq_length

        input_ids = self.all_tokens[s : s + m]
        labels = self.all_tokens[s + 1 : s + m + 1]

        # clone 一下更保险：避免某些测试对 tensor storage/contiguous 做假设
        return {"input_ids": input_ids.clone(), "labels": labels.clone()}

def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    """
    返回一个 DataLoader：迭代一遍就是一个 epoch。
    dataset 的 __getitem__ 已经是定长 tensor，所以默认 collate 就能 stack。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )