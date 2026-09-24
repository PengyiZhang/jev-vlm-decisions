from pathlib import Path

from person_type_a.encoding import assign_letters
from person_type_a.schema import load_scenario, validate

SCEN = Path(__file__).resolve().parent.parent / "scenarios"


def test_builtin_scenarios_valid():
    for name in ("terminal", "apron"):
        sc = load_scenario(SCEN / f"{name}.json")
        validate(sc)
        for q in sc.questions:
            assign_letters(q.effective_options)  # 不抛错


def test_terminal_matches_design_example():
    sc = load_scenario(SCEN / "terminal.json")
    ptype = next(q for q in sc.questions if q.qid == "ptype")
    assert "airport ground staff" in ptype.options
    assert "flight attendant" in ptype.options
    assert "passenger" in ptype.options
    assert set(q.qid for q in sc.questions) == {"ptype", "vest", "luggage"}


def test_apron_is_high_cardinality():
    sc = load_scenario(SCEN / "apron.json")
    ptype = next(q for q in sc.questions if q.qid == "ptype")
    assert 10 <= len(ptype.options) <= 25  # 单字母槽容量内的高基数场景
