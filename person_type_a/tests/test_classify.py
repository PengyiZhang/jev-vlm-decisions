from person_type_a.classify import classify
from person_type_a.engine import CharTokenizer, FakeScorer
from person_type_a.schema import QuestionSpec

VOCAB = 65536


def questions():
    return (
        QuestionSpec("ptype", "choice", "图中人员属于哪种类型？",
                     ("airport ground staff", "flight attendant", "passenger")),
        QuestionSpec("vest", "binary", "是否穿着反光背心？"),
    )


def test_classify_end_to_end_with_fake_scorer():
    # 让 ptype 的 A 槽（airport ground staff）大幅胜出，vest 的 A 槽（no）胜出
    fake = FakeScorer(VOCAB, wins={"ptype": "A", "vest": "A"}, margin=4.0)
    results = classify(fake, CharTokenizer(), "你是个图像分类器", "航站楼", ("制服样式",), questions())
    ptype, vest = results["ptype"], results["vest"]

    assert ptype["top_label"] == "airport ground staff"
    assert ptype["gate"] == "auto" and ptype["top_prob"] > 0.9
    assert ptype["distribution"]["flight attendant"] < 0.1

    assert vest["top_label"] == "no"  # binary 字典序：A=no, B=unclear, C=yes, D=弃权


def test_classify_applies_calibrator_temperature():
    fake = FakeScorer(VOCAB, wins={"ptype": "A"}, margin=0.3)  # 微弱优势
    hot = classify(fake, CharTokenizer(), "y", "s", ("e",), questions()[:1])
    calmed = classify(fake, CharTokenizer(), "y", "s", ("e",), questions()[:1],
                      temperatures={"choice|4": 4.0})
    assert calmed["ptype"]["top_prob"] < hot["ptype"]["top_prob"]


def test_results_carry_abstain_mass():
    fake = FakeScorer(VOCAB, wins={"ptype": "D"}, margin=5.0)  # 弃权槽压倒性胜出
    results = classify(fake, CharTokenizer(), "y", "s", ("e",), questions()[:1])
    assert results["ptype"]["abstain_mass"] > 0.9
    assert results["ptype"]["top_label"] == "__insufficient_evidence__"
