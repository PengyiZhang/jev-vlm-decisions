"""场景与问题配置：规范化排序、弃权槽、K 上限校验。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ABSTAIN_LABEL = "__insufficient_evidence__"
MAX_OPTIONS = 26  # 单字母槽上限（含弃权槽）；超出应走分层（见设计文档）
BINARY_OPTIONS = ("no", "unclear", "yes")  # 字典序稳定


@dataclass(frozen=True)
class QuestionSpec:
    qid: str
    kind: str                       # "choice" | "binary"
    instructions: str
    options: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ()  # 与原始 options 对齐的描述，随排序重排

    def _sorted_pairs(self) -> list[tuple[str, str]]:
        crit = self.criteria or ("",) * len(self.options)
        return sorted(zip(self.options, crit))

    @property
    def effective_options(self) -> tuple[str, ...]:
        """规范顺序（字典序）+ 末位弃权槽，消除选项顺序抖动。"""
        if self.kind == "binary":
            return BINARY_OPTIONS + (ABSTAIN_LABEL,)
        return tuple(label for label, _ in self._sorted_pairs()) + (ABSTAIN_LABEL,)

    @property
    def ordered_criteria(self) -> tuple[str, ...]:
        if self.kind == "binary":
            return ("",) * 4
        return tuple(c for _, c in self._sorted_pairs()) + ("证据不足",)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    system: str
    scene: str
    evidence: tuple[str, ...]
    questions: tuple[QuestionSpec, ...]


def load_scenario(path: str | Path) -> ScenarioConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = tuple(
        QuestionSpec(
            qid=q["qid"], kind=q["kind"], instructions=q["instructions"],
            options=tuple(q.get("options", ())),
            criteria=tuple(q.get("criteria", ())),
        ) for q in raw["questions"]
    )
    return ScenarioConfig(raw["name"], raw["system"], raw["scene"], tuple(raw["evidence"]), questions)


def validate(scenario: ScenarioConfig) -> None:
    for q in scenario.questions:
        if q.kind not in ("choice", "binary"):
            raise ValueError(f"未知问题类型: {q.kind}")
        if len(q.effective_options) > MAX_OPTIONS:
            raise ValueError(
                f"问题 {q.qid} 候选数 {len(q.effective_options)} 超过单字母槽上限 "
                f"{MAX_OPTIONS}（含弃权槽）；请改用分层分类（设计文档·路线 A）"
            )
        if q.kind == "choice" and len(set(q.options)) != len(q.options):
            raise ValueError(f"问题 {q.qid} 存在重复选项")
