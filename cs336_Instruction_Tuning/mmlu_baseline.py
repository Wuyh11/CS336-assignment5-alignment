from __future__ import annotations

import re
from typing import Any


# 预编译正则：匹配 “The correct answer is X”
_RE_TEMPLATE = re.compile(
    r"the\s+correct\s+answer\s+is\s*[:\-]?\s*([ABCD])\b",
    flags=re.IGNORECASE,
)

# 匹配只有一个字母的情况： "A" "(C)" "D."
_RE_SINGLE = re.compile(
    r"^\s*[\(\[\{]?\s*([ABCD])\s*[\)\]\}]?\s*[\.\!\?]?\s*$",
    flags=re.IGNORECASE,
)

# 匹配 "Answer: A" / "答案：B" 之类（更宽松）
_RE_ANSWER_LINE = re.compile(
    r"(?:^|\n)\s*(?:answer|final|答案)\s*[:：]?\s*([ABCD])\b",
    flags=re.IGNORECASE,
)


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    给定 MMLU 样本 + 模型输出，解析预测字母 A/B/C/D；不能解析返回 None。
    """

    def _strip_code_fence(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def _normalize(s: str) -> str:
        # 统一空白，便于做包含判断
        return " ".join(s.strip().split())

    text = _strip_code_fence(model_output or "")

    # 1) 最可靠：按作业指定模板解析
    m = _RE_TEMPLATE.search(text)
    if m:
        return m.group(1).upper()

    # 2) 次可靠：answer/final 行
    m = _RE_ANSWER_LINE.search(text)
    if m:
        return m.group(1).upper()

    # 3) 仅一个字母
    m = _RE_SINGLE.match(text)
    if m:
        return m.group(1).upper()

    # 4) 谨慎兜底：如果文本里作为独立 token 出现的 A/B/C/D 只有一种
    letters = set(re.findall(r"\b([ABCD])\b", text.upper()))
    if len(letters) == 1:
        return next(iter(letters))

    # 5) 利用选项文本兜底：模型输出了选项内容（或非常接近）
    #    思路：看输出里包含哪个 option 文本（忽略大小写与多余空白）。
    options = mmlu_example.get("options")
    if isinstance(options, list) and len(options) == 4:
        norm_out = _normalize(text).lower()
        hits = []
        for i, opt in enumerate(options):
            opt_norm = _normalize(str(opt)).lower()
            if opt_norm and opt_norm in norm_out:
                hits.append(i)
        # 如果只命中一个选项，认为它选择了该选项
        if len(hits) == 1:
            return "ABCD"[hits[0]]

    # 6) 否则无法可靠解析
    return None