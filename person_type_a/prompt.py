"""路线 A 的 Prompt 组装：判定依据放共享指令区；每问一块，答案槽留空。

三个推理引擎统一走 chat 模板：chat_messages 构造消息（图像为 user
content 的 image 块），由 tokenizer.apply_chat_template 渲染成最终文本，
图像占位符由模板注入——不再手工拼接裸文本。
"""
from __future__ import annotations

from dataclasses import dataclass

from .encoding import assign_letters
from .schema import QuestionSpec


@dataclass(frozen=True)
class Slot:
    qid: str
    anchor: str  # 槽位锚文本，如 "答案 1："；读取该位置末 token 的 next-token Logits


def build_system_text(system: str, scene: str, evidence: tuple[str, ...]) -> str:
    """系统区文本：角色指令 + 判定依据 + 场景。"""
    lines = [system]
    if evidence:
        lines.append(f"判定依据：{' / '.join(evidence)}")
    if scene:
        lines.append(f"场景：{scene}")
    return "\n".join(lines)


def _question_lines(k: int, q: QuestionSpec) -> tuple[list[str], str]:
    lines = [f"问题 {k}：{q.instructions}"]
    letters = assign_letters(q.effective_options)
    for label, letter in letters.items():
        idx = q.effective_options.index(label)
        crit = q.ordered_criteria[idx]
        suffix = f"：{crit}" if crit else ""
        lines.append(f"({letter}) {label}{suffix}")
    return lines, f"答案 {k}："


def build_question_text(
    questions: tuple[QuestionSpec, ...], placeholder: str | None = None,
    fill_dummy: bool = False,
) -> tuple[str, tuple[Slot, ...]]:
    """多问题文本 + 槽位表。

    fill_dummy=True 时每问槽位填"超出候选范围的哑字母"（如 4 候选填 E）：
    in-context 锚定"答案槽=字母"格式且不引入候选内容，供 prompt_logprobs
    路径定位读取（空槽会被强指令模型用 <|im_end|>/散文挤掉字母）。
    """
    from .encoding import dummy_letter

    lines: list[str] = []
    slots: list[Slot] = []
    for k, q in enumerate(questions, start=1):
        qlines, anchor = _question_lines(k, q)
        fill = (dummy_letter(len(q.effective_options)) if fill_dummy else placeholder) or ""
        lines.extend(qlines)
        lines.append(anchor + fill)
        slots.append(Slot(q.qid, anchor))
    return "\n".join(lines), tuple(slots)


def build_question_text_single(q: QuestionSpec, k: int = 1) -> str:
    """单问文本（策略一）：只含本问，恰以"答案："结尾交给 1-token 生成。"""
    qlines, _ = _question_lines(k, q)
    return "\n".join(qlines) + "\n答案："


def chat_messages(system_text: str, question_text: str, image=None) -> list[dict]:
    """系统区 + 问题区 → chat 消息；图像对象随 user content 的 image 块传入。"""
    user_content = ([{"type": "image", "image": image}] if image is not None else []) + [
        {"type": "text", "text": question_text}
    ]
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]
