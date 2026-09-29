"""候选答案的 token 表层变体与 LSE 合并。

同一选项在不同锚点下的续写形态不同：prompt 末尾（assistant 首 token）倾向
裸字母 'A'，句中槽位倾向空格形态 ' A'。只取单一形态会静默丢概率，这里对
两种形态的 logits 做 log-sum-exp 作为选项分数。

person_type_a/encoding.letter_token_id 解决的是"选哪个 id"，本模块解决
"多个 id 怎么合并"，二者互补。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def letter_variants(tokenizer, letter: str, context: str = "") -> list[int]:
    """字母的表层变体 id：裸形态 'A'、空格形态 ' A'（上下文解析，失败退回独立编码）。"""
    ids: list[int] = []
    bare = tokenizer.encode(letter, add_special_tokens=False)
    if len(bare) == 1:
        ids.append(bare[0])
    if context:
        full = tokenizer.encode(context + letter, add_special_tokens=False)
        prev = tokenizer.encode(context, add_special_tokens=False)
        if len(full) == len(prev) + 1 and full[: len(prev)] == prev:
            if full[-1] not in ids:
                ids.append(full[-1])
    spaced = tokenizer.encode(" " + letter, add_special_tokens=False)
    if len(spaced) == 1 and spaced[0] not in ids:
        ids.append(spaced[0])
    if not ids:
        raise ValueError(f"字母 {letter} 无可用单 token 形态")
    return ids


def safety_label_variants(tokenizer, label: str) -> list[int]:
    """'Safety:' 之后的标签首 token（' Safe'/' Unsafe'/' Cont'）。

    Controversial 是多 token 词，只取首 token——三个候选首 token 互不碰撞，
    掩码 Softmax 即条件分布。
    """
    context = "Safety:"
    full = tokenizer.encode(context + " " + label, add_special_tokens=False)
    prev = tokenizer.encode(context, add_special_tokens=False)
    if len(full) > len(prev) and full[: len(prev)] == prev:
        return [full[len(prev)]]
    raise ValueError(f"标签 {label} 在 {context!r} 后无法解析出首 token")


def lse_scores(
    logit_row: Sequence[float] | Mapping[int, float],
    variants_per_option: Sequence[Sequence[int]],
) -> list[float]:
    """每选项对其表层变体做 log-sum-exp，返回选项分数（顺序与选项一致）。"""
    flat = [logit_row[i] for vs in variants_per_option for i in vs]
    m = max(flat)
    out = []
    for vs in variants_per_option:
        s = sum(math.exp(logit_row[i] - m) for i in vs)
        out.append(m + math.log(s))
    return out


def check_continuation(tokenizer, prompt: str, surface: str) -> bool:
    """prompt + surface 是否按预期续写（不与 prompt 尾部合并成别的 token）。

    用于启动自检：若 BPE 把 prompt 尾与 surface 合并，读取位的条件分布
    就不是预期的形态，必须拦截。
    """
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    f_ids = tokenizer.encode(prompt + surface, add_special_tokens=False)
    return f_ids[: len(p_ids)] == p_ids and len(f_ids) == len(p_ids) + 1
