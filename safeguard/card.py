"""Native-Slot Answer Card：guard 三行输出的固定槽位重写。

Phase 2 链式读取要 ~16 次前向（484ms）才能覆盖全链；答案卡把变长
输出压成固定槽位，一次前向读全部：

  Safety:                      ← 槽 0：Safe / Unsafe / Controversial（原生锚）
  Violent:                     ← 槽 1-13：13 类各一行，Yes / No
  ...
  Refusal:                     ← 槽 14：Yes / No（仅 assistant 响应样本）

结构基础：槽位 k 的 next-token 分布只依赖锚位 k 之前的 token，
槽间互不干扰——所以全部槽位可以同时留空、单次前向读取。
第一行保留 Route 1 的原生 "Safety:" 锚，零样本即有 97% 起跑线；
槽 1+ 零样本时读不到（模型期望 "Safety: <label>\nCategories:"），
RLCD 训练把"空前缀槽位"的条件化写入模型分布。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .native import SAFETY_LABELS
from .scenario import ALL_CATEGORIES

CARD_SAFETY_ANCHOR = "Safety:"
REFUSAL_ANCHOR = "Refusal:"


@dataclass
class CardResult:
    """答案卡单前向读取的完整输出。"""
    safety_label: str = ""
    safety_probs: dict[str, float] = field(default_factory=dict)
    cat_yes: dict[str, float] = field(default_factory=dict)  # cat -> P(Yes)
    categories: list[str] = field(default_factory=list)      # P(Yes) > P(No) 的类别
    refusal_label: str = ""
    refusal_probs: dict[str, float] = field(default_factory=dict)
    slot_positions: dict[str, int] = field(default_factory=dict)
    error: str = ""


def wants_refusal(messages: list[dict]) -> bool:
    """assistant 响应样本才有 Refusal 槽（对齐 guard 训练配置）。"""
    return bool(messages) and messages[-1].get("role") == "assistant"


def card_lines(include_refusal: bool) -> list[str]:
    lines = [CARD_SAFETY_ANCHOR] + [f"{cat}:" for cat in ALL_CATEGORIES]
    if include_refusal:
        lines.append(REFUSAL_ANCHOR)
    return lines


def card_skeleton(include_refusal: bool) -> str:
    return "\n".join(card_lines(include_refusal))


def card_anchors(include_refusal: bool) -> list[tuple[str, str]]:
    """(槽名, 锚文本)，卡片行序。槽名 cat::<类别名>。

    非末行锚含尾部换行：Qwen pretokenizer 把 ":+\\n" 合并成一个 token
    （正则 " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"），不带换行会在卡内找不到。
    锚末 token（":\\n"）的 next-token 恰是槽位标签。
    """
    lines = card_lines(include_refusal)
    names = (["safety"] + [f"cat::{c}" for c in ALL_CATEGORIES]
             + (["refusal"] if include_refusal else []))
    anchors = [(name, line + "\n") for name, line in zip(names[:-1], lines[:-1])]
    anchors.append((names[-1], lines[-1]))
    return anchors


def render_card_prompt(tokenizer, messages: list[dict], domain_template: str,
                       include_refusal: bool | None = None) -> str:
    """对话渲染 + 空槽答案卡（训练与推理共用同一条件化）。"""
    if include_refusal is None:
        include_refusal = wants_refusal(messages)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, chat_template=domain_template,
        add_generation_prompt=False)
    return rendered + card_skeleton(include_refusal)


def _first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else -1


def safety_token_ids(tokenizer) -> list[int]:
    return [_first_token_id(tokenizer, f" {lbl}") for lbl in SAFETY_LABELS]


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    return _first_token_id(tokenizer, " Yes"), _first_token_id(tokenizer, " No")


def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    n, m = len(haystack), len(needle)
    for i in range(start, n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def locate_slots(tokenizer, input_ids: list[int],
                 include_refusal: bool) -> dict[str, int]:
    """定位每个槽位的读取点（锚末 token 的全局位置）。

    答案卡是 prompt 的尾部后缀，先验证整体 token 对齐（防 BPE 缝隙
    合并），再在卡内按序搜索——不会误匹配对话正文里的同名文本。
    """
    skeleton = card_skeleton(include_refusal)
    skel_ids = tokenizer.encode(skeleton, add_special_tokens=False)
    if input_ids[-len(skel_ids):] != skel_ids:
        raise ValueError("card skeleton not aligned with tokenized tail")

    positions: dict[str, int] = {}
    search_from = 0
    base = len(input_ids) - len(skel_ids)
    for name, anchor in card_anchors(include_refusal):
        a_ids = tokenizer.encode(anchor, add_special_tokens=False)
        idx = find_subsequence(skel_ids, a_ids, search_from)
        if idx < 0:
            raise ValueError(f"anchor {anchor!r} not found in card skeleton")
        positions[name] = base + idx + len(a_ids) - 1
        search_from = idx + len(a_ids)
    return positions


def _masked_probs(logits_row, token_ids: list[int]) -> dict[int, float]:
    import math
    scores = [float(logits_row[i]) for i in token_ids]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return {tid: e / z for tid, e in zip(token_ids, exps)}


def read_card(model, tokenizer, messages: list[dict], domain_template: str,
              include_refusal: bool | None = None) -> CardResult:
    """单次前向读取全部槽位。"""
    import torch

    result = CardResult()
    if include_refusal is None:
        include_refusal = wants_refusal(messages)
    prompt = render_card_prompt(tokenizer, messages, domain_template, include_refusal)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    ids = inputs.input_ids[0].tolist()

    try:
        pos = locate_slots(tokenizer, ids, include_refusal)
    except ValueError as e:
        result.error = str(e)
        return result
    result.slot_positions = pos

    s_ids = safety_token_ids(tokenizer)
    s_probs = _masked_probs(logits[pos["safety"]].float(), s_ids)
    best = max(range(len(s_ids)), key=lambda i: s_probs[s_ids[i]])
    result.safety_label = SAFETY_LABELS[best]
    result.safety_probs = {
        SAFETY_LABELS[i]: round(s_probs[s_ids[i]], 4) for i in range(len(s_ids))
    }

    y_id, n_id = yes_no_token_ids(tokenizer)
    for cat in ALL_CATEGORIES:
        p = _masked_probs(logits[pos[f"cat::{cat}"]].float(), [y_id, n_id])
        result.cat_yes[cat] = round(p[y_id], 4)
    result.categories = [c for c in ALL_CATEGORIES if result.cat_yes[c] > 0.5]

    if include_refusal:
        p = _masked_probs(logits[pos["refusal"]].float(), [y_id, n_id])
        result.refusal_probs = {"Yes": round(p[y_id], 4), "No": round(p[n_id], 4)}
        result.refusal_label = "Yes" if p[y_id] > p[n_id] else "No"
    return result


def temper_probs(probs: list[float], t: float) -> list[float]:
    """温度校准：p^(1/T) 归一。

    与在原始 logits 上除 T 严格等价（softmax 只依赖分数差，
    log p 保留全部差分）；T>0 时 argmax 不变——只重塑置信度，
    不改判定。"""
    if t == 1.0:
        return list(probs)
    exps = [max(p, 1e-12) ** (1.0 / t) for p in probs]
    z = sum(exps)
    return [e / z for e in exps]


def parse_label_text(text: str) -> dict:
    """assistant_label 三行文本 → {safety, categories, refusal}。"""
    out = {"safety": "", "categories": [], "refusal": ""}
    for line in text.strip().split("\n"):
        if line.startswith("Safety:"):
            out["safety"] = line[len("Safety:"):].strip()
        elif line.startswith("Categories:"):
            cats = line[len("Categories:"):].strip()
            out["categories"] = [c.strip() for c in cats.split(",") if c.strip()]
        elif line.startswith("Refusal:"):
            out["refusal"] = line[len("Refusal:"):].strip()
    return out


def gold_slots(label_text: str, include_refusal: bool) -> dict | None:
    """答案卡金标。safety 标签非法或 refusal 缺失时返回 None（跳过样本）。"""
    parsed = parse_label_text(label_text)
    if parsed["safety"] not in SAFETY_LABELS:
        return None
    if include_refusal and parsed["refusal"] not in ("Yes", "No"):
        return None
    return {
        "safety": parsed["safety"],
        "cats": {c: (1 if c in parsed["categories"] else 0) for c in ALL_CATEGORIES},
        "refusal": parsed["refusal"],
    }
