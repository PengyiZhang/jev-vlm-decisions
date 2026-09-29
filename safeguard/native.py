"""Route 1 原生锚定：guard chat template 原文 + 强制 "Safety:" 前缀。

微调模型的输出三行共享 "Safety: " 前缀，把该前缀并入 prefill 后，
末位的 next-token 分布就是模型自身的 P(Safe/Unsafe/Controversial | 对话)。
零格式偏移——这是对微调分布最忠实的单次前向读取。

只能覆盖 safety 三分类：Categories 是变长逗号列表，单一列表位置读不出
13 个独立概率；Refusal 在第三行，需先确定前两行内容。结构性限制，
不是实现细节。
"""
from __future__ import annotations

from pathlib import Path

SAFETY_ANCHOR = "Safety:"
SAFETY_LABELS = ("Controversial", "Safe", "Unsafe")  # 与 scenario 字典序一致

DEFAULT_DOMAIN_TEMPLATE = Path(
    "/data0/zhangpengyi/LLMs/Qwen3Guard-Gen-0.6B/chat_template.domain.jinja"
)


def load_domain_template(path: str | Path | None = None) -> str:
    return Path(path or DEFAULT_DOMAIN_TEMPLATE).read_text(encoding="utf-8")


def build_native_prompt(
    tokenizer, messages: list[dict], domain_template: str
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        chat_template=domain_template,
        add_generation_prompt=False,
    )
    return rendered + SAFETY_ANCHOR
