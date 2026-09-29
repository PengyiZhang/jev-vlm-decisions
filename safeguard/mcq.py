"""Route 2 字母槽 MCQ prompt 构建。

2a（每问独立）：标准 MCQ——英文、`A. option` 选项、`Answer:` 锚在 user
   消息末尾，assistant 前导（空 think 块）之后读末位 logits。这是
   person_type_a CIFAR-10 实验证实的零样本有效形态（1.0% -> 86.5%）。

2b（多问合一）：user 消息放对话与全部问题；assistant 前导后接"答题卡"
   `1. E\n2. E\n…`，每槽填候选（A-D）之外的哑字母 E——in-context 锚定
   "槽=字母"格式且不引入候选内容（person_type_a/prompt.fill_dummy 思路）。
   槽读取位用 offset 定位：哑字母前的最后一个 token。单次前向读全部槽，
   0 解码步，这是 jev 的终极形态。
"""
from __future__ import annotations

from person_type_a.encoding import LETTERS
from .scenario import SAFETY_POLICY, GuardQuestion

DUMMY_LETTER = "E"  # 候选 A-D 之外的第一个字母，仅作槽位格式锚定

SYSTEM_TEXT = (
    "You are a content safety classifier for a Hong Kong elderly-care platform. "
    + SAFETY_POLICY
    + " Answer every question with a single letter."
)


def conversation_text(messages: list[dict]) -> str:
    """guard 原生对话格式（与 chat_template.domain 一致的 USER/ASSISTANT 前缀）。"""
    lines = []
    for m in messages:
        speaker = "USER" if m["role"] in ("user", "system") else "ASSISTANT"
        lines.append(f"{speaker}: {m['content']}")
    return "\n\n".join(lines)


def qwen_chat_markup(system: str, user: str, assistant_suffix: str = "") -> str:
    """手工 Qwen3 chat 标记：模型内嵌模板是 guard 专用的，不能用于任意 MCQ。

    assistant 前导带空 think 块，与 guard 微调分布一致。
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_suffix}"
    )


def question_block(k: int, q: GuardQuestion) -> str:
    lines = [f"Question {k}: {q.instructions}"]
    for letter, opt in zip(LETTERS, q.options):
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def build_2a_prompt(messages: list[dict], questions: tuple[GuardQuestion, ...], idx: int) -> str:
    """第 idx 问的独立 MCQ prompt，以 `Answer:` 收尾。"""
    user = "\n".join([
        "Conversation:", "",
        conversation_text(messages), "",
        question_block(idx + 1, questions[idx]), "",
        "Answer:",
    ])
    return qwen_chat_markup(SYSTEM_TEXT, user)


def build_2b_prompt(
    messages: list[dict], questions: tuple[GuardQuestion, ...], fill: str = DUMMY_LETTER
) -> tuple[str, list[int]]:
    """多问合一 prompt + 每槽答案起始字符位置（供 offset 定位读取位）。

    fill=DUMMY_LETTER：槽内填候选外的哑字母，锚定"槽=字母"格式
    （person_type_a 的 fill_dummy 思路）；fill=""：空槽消融，
    排除"模型抄哑字母"这一混淆因素。两种形态下读取位规则一致：
    答案起始字符之前的最后一个 token。
    """
    user_lines = ["Conversation:", "", conversation_text(messages), ""]
    for k, q in enumerate(questions, start=1):
        user_lines.append(question_block(k, q))
        user_lines.append("")
    user = "\n".join(user_lines)

    prefix = f". {fill}" if fill else ". "
    sheet = "\n".join(f"{k}{prefix}" for k in range(1, len(questions) + 1))
    prompt = qwen_chat_markup(SYSTEM_TEXT, user, sheet)

    base = prompt.index(sheet)  # 答题卡只出现一次，顺序扫描各槽
    answer_positions: list[int] = []
    cursor = 0
    for k in range(1, len(questions) + 1):
        line = f"{k}{prefix}"
        off = prompt.index(line, base + cursor)
        answer_positions.append(off + len(line) - 1)
        cursor = off - base + len(line)
    return prompt, answer_positions


def read_positions(offsets: list[tuple[int, int]], answer_positions: list[int]) -> list[int]:
    """槽读取位 = 答案起始字符之前的最后一个 token。

    哑字母形态下，哑字母可能与其前导空格合并成 ' E'（offset 起点在空格
    字符）；空槽形态下，行尾空格可能与换行合并成 ' \\n'。统一取
    b <= 答案起始字符位的最后一个 token，读取位的 next-token 分布
    即该槽的答案分布。
    """
    out = []
    for cp in answer_positions:
        idx = max(i for i, (a, b) in enumerate(offsets) if b <= cp)
        out.append(idx)
    return out
