"""字母槽编码：候选 → 单 token 字母。K>26 已由 schema.validate 拦截。"""
from __future__ import annotations

from typing import Protocol

LETTERS = [chr(ord("A") + i) for i in range(26)]


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def assign_letters(options: tuple[str, ...]) -> dict[str, str]:
    """按给定顺序（即 effective_options 的规范顺序）分配 A、B、C…。"""
    if len(options) > len(LETTERS):
        raise ValueError(f"候选数 {len(options)} 超过字母槽 {len(LETTERS)}")
    return dict(zip(options, LETTERS))


def check_letters_single_token(tokenizer: TokenizerLike) -> list[str]:
    """返回不是单 token 的字母。真实模型上线前必须为空（设计文档·词表实测）。"""
    return [l for l in LETTERS if len(tokenizer.encode(l, add_special_tokens=False)) != 1]


def letter_token_id(tokenizer: TokenizerLike, letter: str, context: str = "：") -> int:
    """按上下文后缀解析字母 token id。

    模型在"："之后的真实续写形态可能是带前缀的变体（SentencePiece 的
    ▁A），裸 encode("A") 的 id 在 top-K logprobs 里可能根本不出现。
    取 encode(context + letter) 相对 encode(context) 的增量；无法解析时
    退回裸形态。
    """
    if context:
        full = tokenizer.encode(context + letter, add_special_tokens=False)
        prev = tokenizer.encode(context, add_special_tokens=False)
        if len(full) > len(prev) and full[: len(prev)] == prev:
            suffix = full[len(prev):]
            if len(suffix) == 1:
                return suffix[0]
    ids = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"字母 {letter} 非单 token: {ids}")
    return ids[0]
