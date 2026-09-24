from person_type_a.prompt import (
    build_question_text, build_question_text_single, build_system_text, chat_messages,
)
from person_type_a.schema import QuestionSpec


def scenario_questions():
    return (
        QuestionSpec("ptype", "choice", "图中人员属于哪种类型？",
                     ("airport ground staff", "flight attendant", "passenger"),
                     ("地面作业制服、反光背心", "航司制服丝巾", "普通旅客装束")),
        QuestionSpec("vest", "binary", "是否穿着反光背心？"),
    )


def test_system_text_contains_scene_and_evidence():
    text = build_system_text("你是个图像分类器", "航站楼到港层", ("制服样式", "反光背心"))
    assert "图像分类器" in text and "制服样式" in text and "航站楼到港层" in text


def test_question_text_numbered_slots_unique():
    text, slots = build_question_text(scenario_questions())
    assert [s.anchor for s in slots] == ["答案 1：", "答案 2："]
    for s in slots:
        assert text.count(s.anchor) == 1  # 锚文本在全文唯一


def test_options_rendered_with_letters_and_criteria():
    text, _ = build_question_text(scenario_questions())
    assert "(A) airport ground staff：地面作业制服、反光背心" in text
    assert "(D) __insufficient_evidence__：证据不足" in text  # ptype 4 槽
    assert "(A) no" in text  # binary 字典序 A=no


def test_slot_order_follows_question_blocks():
    text, _ = build_question_text(scenario_questions())
    assert text.index("答案 1：") < text.index("问题 2：")


def test_placeholder_fills_slots():
    text, slots = build_question_text(scenario_questions(), placeholder="？")
    assert "答案 1：？" in text and "答案 2：？" in text
    for s in slots:  # 锚本身（不含占位符）仍全文唯一
        assert text.count(s.anchor) == 1


def test_single_question_text_ends_at_anchor():
    text = build_question_text_single(scenario_questions()[0])
    assert text.endswith("答案：") and not text.endswith("答案：\n")
    assert "airport ground staff" in text
    assert "是否穿着反光背心" not in text  # 不含别的问题


def test_chat_messages_structure_with_image():
    img = object()
    msgs = chat_messages("sys", "qtext", img)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1]["content"][0] == {"type": "image", "image": img}
    assert msgs[1]["content"][1] == {"type": "text", "text": "qtext"}


def test_chat_messages_without_image():
    msgs = chat_messages("sys", "qtext")
    assert all(c.get("type") != "image" for c in msgs[1]["content"])
