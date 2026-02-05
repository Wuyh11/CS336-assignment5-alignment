from __future__ import annotations

import re
from typing import Optional

# 说明：
# GSM8K 要求：把模型输出里“最后出现的数字”当作预测答案。
# 这里的“数字”我们用正则抓取：
#   - 可选负号
#   - 整数或小数
#   - 支持千分位逗号（如 12,345）
#   - 不把像 3/4 这种分数当作一个数字（GSM8K 标准答案基本是整数）
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def _strip_code_fence(s: str) -> str:
    """轻量去掉最外层 ```...```，避免代码块里的内容干扰，但不强制依赖。"""
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_gsm8k_response(model_output: str) -> Optional[str]:
    """
    按 GSM8K baseline 要求：
      - 找到 model_output 中最后出现的一个“数字串”
      - 返回该数字串的规范化形式（去掉逗号）
      - 如果找不到任何数字，返回 None
    返回类型是 str（与测试签名一致）
    """
    text = _strip_code_fence(model_output)

    matches = list(_NUM_RE.finditer(text))
    if not matches:
        return None

    last = matches[-1].group(0)

    # 去掉千分位逗号
    last = last.replace(",", "").strip()

    # 一些边缘情况：正则可能抓到 "-" 或 "." 这类不完整的数字（理论上很少）
    # 做个安全检查：至少得包含一个数字字符
    if not any(ch.isdigit() for ch in last):
        return None

    return last