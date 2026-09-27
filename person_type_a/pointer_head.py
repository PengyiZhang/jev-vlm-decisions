"""指针头：交叉注意力式选项打分，绕过字母映射。

kev 的架构（model.py:123-129）移植：每个选项末尾 </opt> 位置的隐状态作为 key，
<decide> 位置的隐状态作为 query，投影后做点积打分，softmax 得选项分布。

与字母槽的本质区别：模型不需要学"类名→字母"的人工编码，只需要用
Transformer 天然的注意力机制做"视觉表示 ↔ 选项表示"的对齐。
"""
from __future__ import annotations

from __future__ import annotations

import torch
import torch.nn as nn

# torch 惰性导入兼容：非训练环境仅使用 PointerHead 的类型签名


class PointerHead(nn.Module):
    """单头交叉注意力指针：query 对各选项 key 打分。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, option_hiddens: torch.Tensor,
                decide_hidden: torch.Tensor) -> torch.Tensor:
        """option_hiddens: [batch, n_options, hidden]
        decide_hidden: [batch, hidden]
        Returns: [batch, n_options] 概率分布
        """
        keys = self.W_k(option_hiddens)                     # [b, k, d]
        query = self.W_q(decide_hidden)                     # [b, d]
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # [b, k]
        return torch.softmax(scores, dim=-1)


def build_pointer_prompt(instructions: str, options: list[str]) -> str:
    """构造指针头专用的 prompt：选项后跟 </opt>，末尾 <decide>。

    与字母槽布局不同：不分配字母，不写"答案："锚。
    """
    lines = [instructions]
    for opt in options:
        lines.append(f"{opt} </opt>")
    lines.append("<decide>")
    return "\n".join(lines)


def find_marker_positions(token_ids: list[int],
                          marker_ids: list[int],
                          start: int = 0) -> list[int]:
    """在 token 序列中找出所有 marker 出现的末位下标。"""
    positions = []
    n = len(marker_ids)
    for i in range(start, len(token_ids) - n + 1):
        if token_ids[i:i + n] == marker_ids:
            positions.append(i + n - 1)
    return positions
