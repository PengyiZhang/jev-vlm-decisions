from person_type_a.prompt import build_layout
from person_type_a.schema import QuestionSpec


def scenario_questions():
    return (
        QuestionSpec("ptype", "choice", "图中人员属于哪种类型？",
                     ("airport ground staff", "flight attendant", "passenger"),
                     ("地面作业制服、反光背心", "航司制服丝巾", "普通旅客装束")),
        QuestionSpec("vest", "binary", "是否穿着反光背心？"),
    )


def test_layout_contains_shared_evidence_and_scene():
    lay = build_layout("你是个图像分类器", "航站楼到港层", ("制服样式", "反光背心"), scenario_questions())
    assert "图像分类器" in lay.text
    assert "制服样式" in lay.text and "反光背心" in lay.text
    assert "航站楼到港层" in lay.text
    assert "<image>" in lay.text


def test_slots_are_numbered_and_unique():
    lay = build_layout("y", "s", ("e",), scenario_questions())
    assert [s.anchor for s in lay.slots] == ["答案 1：", "答案 2："]
    for s in lay.slots:
        assert lay.text.count(s.anchor) == 1  # 锚文本在全文唯一


def test_options_rendered_with_letters_and_criteria():
    lay = build_layout("y", "s", ("e",), scenario_questions())
    assert "(A) airport ground staff：地面作业制服、反光背心" in lay.text
    assert "(D) __insufficient_evidence__：证据不足" in lay.text  # ptype 4 槽
    assert "(A) no" in lay.text  # binary 字典序 A=no


def test_slot_follows_its_question_block():
    lay = build_layout("y", "s", ("e",), scenario_questions())
    assert lay.text.index("答案 1：") < lay.text.index("问题 2：")


def test_single_question_prompt_contains_only_its_question():
    from person_type_a.prompt import build_single_question_prompt
    q = scenario_questions()[0]
    text = build_single_question_prompt("你是个图像分类器", "航站楼", ("制服样式",), q)
    assert text.endswith("答案：")  # 恰好止于锚，无尾换行
    assert "airport ground staff" in text
    assert "问题 2" not in text and "是否穿着反光背心" not in text  # 不含别的问题
    assert "制服样式" in text  # 共享指令区保留


def test_placeholder_layout_fills_slots():
    from person_type_a.prompt import build_layout as bl
    lay = bl("y", "s", ("e",), scenario_questions(), placeholder="？")
    assert "答案 1：？" in lay.text and "答案 2：？" in lay.text
    for s in lay.slots:  # 锚本身（不含占位符）仍全文唯一
        assert lay.text.count(s.anchor) == 1


def test_layout_with_custom_image_token():
    from person_type_a.prompt import build_layout as bl
    lay = bl("y", "s", ("e",), scenario_questions(), image_token="<|image_pad|>")
    assert "<|image_pad|>" in lay.text and "<image>" not in lay.text


def test_single_question_prompt_with_custom_image_token():
    from person_type_a.prompt import build_single_question_prompt as sp
    text = sp("y", "s", ("e",), scenario_questions()[0], image_token="<|image|>")
    assert "<|image|>" in text and "<image>" not in text


def test_single_question_prompt_ends_exactly_at_anchor():
    from person_type_a.prompt import build_single_question_prompt as sp
    text = sp("y", "s", ("e",), scenario_questions()[0])
    assert text.endswith("答案：") and not text.endswith("答案：\n")


def test_build_system_text_has_no_image_token():
    from person_type_a.prompt import build_system_text
    text = build_system_text("你是个图像分类器", "航站楼", ("制服样式",))
    assert "图像分类器" in text and "制服样式" in text and "航站楼" in text
    assert "<image>" not in text  # 图像由 chat template 的 image content 注入


def test_build_question_text_returns_slots_without_header():
    from person_type_a.prompt import build_question_text
    text, slots = build_question_text(scenario_questions())
    assert "答案 1：" in text and "答案 2：" in text
    assert "判定依据" not in text  # header 内容不混入问题区
    assert [s.qid for s in slots] == ["ptype", "vest"]
