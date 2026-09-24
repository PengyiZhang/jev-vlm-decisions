"""打分引擎：ClassifyTask + Scorer 协议 + 测试用 Fake；真实适配器见各 *_scorer.py。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .encoding import letter_token_id
from .schema import QuestionSpec

SlotRow = Mapping[int, float]  # 槽位 Logits 行：全词表 list 或 vLLM top-K 稀疏 dict


@dataclass(frozen=True)
class ClassifyTask:
    """scorer 的结构化输入：各策略据此自选 prompt 形态（多问单发 / 每问独立 / 占位符）。"""
    system: str
    scene: str
    evidence: tuple[str, ...]
    questions: tuple[QuestionSpec, ...]


class CharTokenizer:
    """字符级假分词器：id = ord(char)，用于全链路纯逻辑测试。"""
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class Scorer(Protocol):
    def slot_logits(self, task: ClassifyTask, image=None) -> dict[str, SlotRow]:
        """返回 {qid: 槽位处 Logits 行}。"""
        ...


class FakeScorer:
    """按 wins 指定每个问题的胜出字母，构造槽位 Logits 行。"""

    _LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, vocab_size: int, wins: dict[str, str], margin: float = 3.0):
        self.vocab_size = vocab_size
        self.wins = wins
        self.margin = margin

    def slot_logits(self, task: ClassifyTask, image=None) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for q in task.questions:
            row = [0.0] * self.vocab_size
            win_letter = self.wins[q.qid]
            for j, ch in enumerate(self._LETTERS):
                row[ord(ch)] = self.margin if ch == win_letter else 0.1 * ((j % 3) + 1)
            out[q.qid] = row
        return out


def slot_letter_ids(
    tokenizer, task: ClassifyTask, letters_by_qid: dict[str, dict[str, str]]
) -> dict[str, list[int]]:
    """每个问题的有效字母（顺序与 effective_options 对齐）→ token id 列表。"""
    out = {}
    for q in task.questions:
        letters = letters_by_qid[q.qid]
        out[q.qid] = [letter_token_id(tokenizer, l) for l in letters.values()]
    return out
