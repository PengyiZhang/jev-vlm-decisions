from person_type_a.json_baseline import build_json_prompt, parse_model_json


def test_parse_model_json_plain():
    assert parse_model_json('{"ptype": "passenger"}') == {"ptype": "passenger"}


def test_parse_model_json_fenced():
    text = '好的，答案是：\n```json\n{"ptype": "passenger", "vest": "yes"}\n```\n以上。'
    assert parse_model_json(text) == {"ptype": "passenger", "vest": "yes"}


def test_parse_model_json_garbage_returns_none():
    assert parse_model_json("我认为是乘客。") is None


def test_build_json_prompt_lists_all_qids_and_options():
    from person_type_a.schema import QuestionSpec
    qs = (QuestionSpec("ptype", "choice", "类型？", ("a", "b")),
          QuestionSpec("vest", "binary", "背心？"))
    prompt = build_json_prompt(qs)
    assert "ptype" in prompt and "vest" in prompt
    assert "__insufficient_evidence__" in prompt  # 弃权槽也在候选列表里
