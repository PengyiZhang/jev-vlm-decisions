"""槽位读取：候选 Logits → 温度 Softmax → 分布与三级门控。纯标准库。"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .schema import ABSTAIN_LABEL

GATE_HI, GATE_LO = 0.90, 0.60
LOW_SCORE = -1e9  # 稀疏行（vLLM top-K logprobs）未覆盖的候选字母的地板分


def masked_softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    m = max(s / temperature for s in scores)
    exps = [math.exp(s / temperature - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def read_slot(vocab_logits: list[float] | Mapping[int, float], token_ids: list[int]) -> list[float]:
    """从槽位 Logits 行中抽取候选字母槽分数。

    支持两种行形态：transformers 的全词表 list，与 vLLM top-K 的稀疏 dict。
    稀疏行未覆盖的字母给 LOW_SCORE（概率归零，不静默丢失）。
    """
    if isinstance(vocab_logits, Mapping):
        return [vocab_logits.get(i, LOW_SCORE) for i in token_ids]
    return [vocab_logits[i] for i in token_ids]


def decide(probs: list[float], labels: tuple[str, ...], hi: float = GATE_HI, lo: float = GATE_LO) -> dict:
    top = max(range(len(probs)), key=lambda i: probs[i])
    conf = probs[top]
    if labels[top] == ABSTAIN_LABEL:
        gate = "human"  # 弃权胜出 = 模型声明证据不足，语义上不可自动执行
    else:
        gate = "auto" if conf >= hi else ("review" if conf >= lo else "human")
    return {
        "top_label": labels[top],
        "top_prob": round(conf, 4),
        "abstain_mass": round(probs[labels.index(ABSTAIN_LABEL)], 4) if ABSTAIN_LABEL in labels else 0.0,
        "gate": gate,
    }
