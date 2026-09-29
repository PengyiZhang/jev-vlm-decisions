"""Phase 2 链式读取：顺着微调分布逐步执行三行输出，每步读概率。

Route 1 只覆盖 Safety 行；Categories 变长列表和 Refusal 链尾条件读不到。
链式读取把模型的训练格式当成程序逐步执行：
  Forward 1: ...对话... "Safety: "         → P(Safe/Unsafe/Controversial)
  Forward 2: ...Safety: Unsafe\nCategories:  → P(首类别 / 换行)
  Forward 3: ...Categories: Violent,        → P(下一类别 / 换行)
  Forward 4: ...\nRefusal:                  → P(Yes/No)

每个 Forward 共享对话前缀的 KV Cache，新增 suffix 计算量极少。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .native import SAFETY_ANCHOR, SAFETY_LABELS, build_native_prompt
from .scenario import ALL_CATEGORIES

CATEGORIES_ANCHOR = "\nCategories:"
REFUSAL_ANCHOR = "\nRefusal:"
YES_TOKENS = ("Yes", " yes", " Yes")
NO_TOKENS = ("No", " no", " No")


@dataclass
class ChainResult:
    """链式读取的完整输出。"""
    safety_label: str = ""
    safety_probs: dict[str, float] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    category_probs: dict[str, float] = field(default_factory=dict)
    refusal_label: str = ""
    refusal_probs: dict[str, float] = field(default_factory=dict)
    forward_count: int = 0
    error: str = ""


def _read_probs(logits_row, token_ids: list[int]) -> dict[int, float]:
    """从一条全词表 Logits 行中取候选分数并 Softmax。"""
    import math
    scores = [logits_row[i] for i in token_ids]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return {tid: e / z for tid, e in zip(token_ids, exps)}


def _first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else -1


def _match_category(tokenizer, logits_row, categories: list[str]) -> tuple[str, float]:
    """读 Logits 行，返回最可能的类别名和概率。"""
    cat_ids = [_first_token_id(tokenizer, cat) for cat in categories]
    probs = _read_probs(logits_row, cat_ids)
    best_idx = max(range(len(cat_ids)), key=lambda i: probs[cat_ids[i]])
    return categories[best_idx], probs[cat_ids[best_idx]]


def chain_read(
    model,
    tokenizer,
    messages: list[dict],
    domain_template: str,
    max_categories: int = 13,
    include_refusal: bool = True,
) -> ChainResult:
    """链式读取三行输出，每步一次 forward + 概率读取。"""
    import torch

    result = ChainResult()

    # ── Forward 1: Safety 行（与 Route 1 相同） ──
    prompt_1 = build_native_prompt(tokenizer, messages, domain_template)
    inputs_1 = tokenizer(prompt_1, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs_1).logits[0]
    result.forward_count += 1

    safety_ids = [_first_token_id(tokenizer, f" {lbl}") for lbl in SAFETY_LABELS]
    s_probs = _read_probs(logits[-1].float(), safety_ids)
    best_idx = max(range(len(safety_ids)), key=lambda i: s_probs[safety_ids[i]]) if safety_ids else 0
    result.safety_label = SAFETY_LABELS[best_idx]
    result.safety_probs = {
        SAFETY_LABELS[i]: round(s_probs[safety_ids[i]], 4) for i in range(len(safety_ids))
    }

    # ── Forward 2+: Categories 链式读取 ──
    base_prompt = prompt_1  # ending with "Safety:"
    safety_suffix = f" {result.safety_label}{CATEGORIES_ANCHOR} "
    ctx_ids = inputs_1.input_ids[0].tolist()
    suffix_ids = tokenizer.encode(safety_suffix, add_special_tokens=False)
    all_ids = ctx_ids + suffix_ids
    full_ids = tokenizer.decode(all_ids)
    # 重新 tokenize 确保 token 边界一致
    input_ids = tokenizer(full_ids, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**input_ids).logits[0]
    result.forward_count += 1

    remaining = list(ALL_CATEGORIES)
    found: list[str] = []

    for _ in range(max_categories):
        if not remaining:
            break
        cat, prob = _match_category(tokenizer, logits[-1].float(), remaining)
        if prob < 0.05:  # 低于阈值视为列表结束
            break
        found.append(cat)
        result.category_probs[cat] = round(prob, 4)
        remaining.remove(cat)

        # 追加已识别类别，读下一位置
        chain_text = full_ids + cat + ", "
        input_ids = tokenizer(chain_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**input_ids).logits[0]
        result.forward_count += 1

    result.categories = found

    # ── Final Forward: Refusal ──
    if include_refusal:
        cat_list = ", ".join(found) if found else ""
        refusal_prompt = (
            build_native_prompt(tokenizer, messages, domain_template)
            + f" {result.safety_label}{CATEGORIES_ANCHOR} {cat_list}{REFUSAL_ANCHOR} "
        )
        input_ids = tokenizer(refusal_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**input_ids).logits[0]
        result.forward_count += 1

        yes_id = _first_token_id(tokenizer, " Yes")
        no_id = _first_token_id(tokenizer, " No")
        r_probs = _read_probs(logits[-1].float(), [yes_id, no_id])
        result.refusal_probs = {"Yes": round(r_probs[yes_id], 4), "No": round(r_probs[no_id], 4)}
        result.refusal_label = "Yes" if r_probs[yes_id] > r_probs[no_id] else "No"

    return result
