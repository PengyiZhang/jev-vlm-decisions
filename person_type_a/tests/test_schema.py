import json
from pathlib import Path

from person_type_a.schema import (
    ABSTAIN_LABEL, MAX_OPTIONS, QuestionSpec, ScenarioConfig, load_scenario, validate,
)


def make_question(**kw):
    base = dict(qid="ptype", kind="choice", instructions="图中人员属于哪种类型？",
                options=("passenger", "flight attendant", "airport ground staff"))
    base.update(kw)
    return QuestionSpec(**base)


def test_effective_options_sorted_with_abstain():
    q = make_question()
    eff = q.effective_options
    # 规范顺序 = 字典序，末位恒为弃权槽
    assert eff == tuple(sorted(q.options)) + (ABSTAIN_LABEL,)


def test_binary_kind_has_fixed_options():
    q = make_question(qid="vest", kind="binary", instructions="是否穿着反光背心？", options=())
    assert q.effective_options == ("no", "unclear", "yes", ABSTAIN_LABEL)


def test_validate_rejects_oversized_choice():
    big = make_question(options=tuple(f"class_{i}" for i in range(MAX_OPTIONS)))
    try:
        validate(ScenarioConfig("y", "t", "航站楼", ("制服样式",), (big,)))
        raise AssertionError("should reject")
    except ValueError as e:
        assert "26" in str(e)


def test_load_scenario_roundtrip(tmp_path: Path):
    raw = {
        "name": "terminal", "scene": "航站楼到港层",
        "system": "你是机场监控的人员分类器。只输出选项字母，不解释。",
        "evidence": ["制服样式", "反光背心", "工牌"],
        "questions": [
            {"qid": "ptype", "kind": "choice", "instructions": "人员类型？",
             "options": ["passenger", "flight attendant"], "criteria": ["普通旅客装束", "航司制服"]},
            {"qid": "vest", "kind": "binary", "instructions": "是否穿反光背心？"},
        ],
    }
    p = tmp_path / "terminal.json"
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    sc = load_scenario(p)
    assert sc.name == "terminal"
    assert len(sc.questions) == 2
    validate(sc)  # 不抛错
