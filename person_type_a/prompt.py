"""路线 A 的 Prompt 组装：判定依据放共享指令区；每问一块，答案槽留空。

两种组装形态：
- 原始文本布局（build_layout / build_single_question_prompt）：vLLM 等
  直接吃裸文本的引擎使用，图像占位符内联在文本里。
- chat 部件（build_system_text / build_question_text）：transformers 的
  apply_chat_template 路径使用，图像由消息里的 image content 注入，
  文本侧不含任何图像占位符。
"""
from __future__ import annotations

from dataclasses import dataclass

from .encoding import assign_letters
from .schema import QuestionSpec


@dataclass(frozen=True)
class Slot:
    qid: str
    anchor: str  # 槽位锚文本，如 "答案 1："；读取该位置末 token 的 next-token Logits


@dataclass(frozen=True)
class PromptLayout:
    text: str
    slots: tuple[Slot, ...]


def build_system_text(system: str, scene: str, evidence: tuple[str, ...]) -> str:
    """系统区文本：角色指令 + 判定依据 + 场景。不含图像占位符。"""
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
) -> tuple[str, tuple[Slot, ...]]:
    """问题区文本 + 槽位表。placeholder 非空时（策略二b）槽位填中性占位符。"""
    lines: list[str] = []
    slots: list[Slot] = []
    for k, q in enumerate(questions, start=1):
        qlines, anchor = _question_lines(k, q)
        lines.extend(qlines)
        lines.append(anchor + (placeholder or ""))
        slots.append(Slot(q.qid, anchor))
    return "\n".join(lines), tuple(slots)


def build_layout(
    system: str,
    scene: str,
    evidence: tuple[str, ...],
    questions: tuple[QuestionSpec, ...],
    placeholder: str | None = None,
    image_token: str = "<image>",
) -> PromptLayout:
    """原始文本多问题布局（vLLM 路径）：系统区 + 内联图像占位符 + 问题区。"""
    qtext, slots = build_question_text(questions, placeholder)
    text = build_system_text(system, scene, evidence) + "\n" + image_token + "\n" + qtext + "\n"
    return PromptLayout(text, slots)


def build_single_question_prompt(
    system: str,
    scene: str,
    evidence: tuple[str, ...],
    q: QuestionSpec,
    k: int = 1,
    image_token: str = "<image>",
) -> str:
    """策略一：每问独立 prompt（vLLM 路径），以"答案："结尾交给 1-token 生成。"""
    lines = build_system_text(system, scene, evidence).split("\n") + [image_token]
    qlines, _ = _question_lines(k, q)
    lines.extend(qlines)
    lines.append("答案：")
    # 不加尾换行：生成位置的分布必须恰好是"答案："之后的续写
    return "\n".join(lines)
