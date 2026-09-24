import sys
from types import SimpleNamespace

import pytest

from person_type_a.engine import CharTokenizer
from person_type_a.schema import QuestionSpec
from person_type_a.transformers_scorer import TransformersScorer, find_subsequence


def test_find_subsequence_finds_last_occurrence():
    hay, needle = [1, 2, 3, 9, 1, 2, 3], [1, 2, 3]
    assert find_subsequence(hay, needle) == 6  # 末次出现的末元素下标


def test_find_subsequence_not_found():
    assert find_subsequence([1, 2], [7, 8]) == -1


def test_constructor_does_not_import_torch():
    # 实例化只存配置；torch 应在 load() 时才被导入
    s = TransformersScorer("fake-model")
    assert s.model_id == "fake-model" and s._model is None


def test_require_deps_raises_helpfully_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    s = TransformersScorer("fake-model")
    with pytest.raises(ImportError, match="torch"):
        s._require_deps(raise_if_missing=True)


def test_verify_anchors_catches_broken_anchor():
    class BrokenProcessor(FakeChatProcessor):
        def apply_chat_template(self, messages, **kw):
            # 模板丢失问题区文本：锚无处可寻
            return {"input_ids": [[ord(c) for c in "<s>hello"]]}

    s = TransformersScorer.__new__(TransformersScorer)
    s._tokenizer = CharTokenizer()
    s._processor = BrokenProcessor()
    from person_type_a.engine import ClassifyTask

    task = ClassifyTask("y", "s", ("e",), (QuestionSpec("ptype", "choice", "q？", ("a", "b")),))
    assert s.verify_anchors(task) is False


class VocabTokenizer:
    """带词表探测能力的假分词器。"""
    def __init__(self, vocab: dict):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok)


def test_resolve_image_token_prefers_processor_attr():
    from person_type_a.transformers_scorer import resolve_image_token
    proc = type("P", (), {"image_token": "<|image|>"})()
    tok = VocabTokenizer({"<|image_pad|>": 1, "<|image|>": 2})
    assert resolve_image_token(proc, tok) == "<|image|>"


def test_resolve_image_token_wraps_qwen_pad_with_vision_markers():
    from person_type_a.transformers_scorer import resolve_image_token
    proc = type("P", (), {"image_token": "<|image_pad|>"})()
    tok = VocabTokenizer({"<|image_pad|>": 1, "<|vision_start|>": 2, "<|vision_end|>": 3})
    assert resolve_image_token(proc, tok) == "<|vision_start|><|image_pad|><|vision_end|>"


def test_resolve_image_token_falls_back_to_vocab_scan():
    from person_type_a.transformers_scorer import resolve_image_token
    proc = type("P", (), {})()  # processor 无 image_token 属性
    tok = VocabTokenizer({"<|image|>": 5})
    assert resolve_image_token(proc, tok) == "<|image|>"


class FakeChatProcessor:
    """仿真 processor.apply_chat_template：拼消息文本并按字符转 id。"""

    def apply_chat_template(self, messages, **kw):
        parts = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                for c in content:
                    parts.append("[IMG]" if c["type"] == "image" else c["text"])
            else:
                parts.append(content)
        text = "<s>" + "\n".join(parts)
        return {"input_ids": [[ord(ch) for ch in text]]}


def _chat_scorer():
    from person_type_a.transformers_scorer import TransformersScorer
    s = TransformersScorer.__new__(TransformersScorer)
    s._tokenizer = CharTokenizer()
    s._processor = FakeChatProcessor()
    return s


def test_build_messages_structure():
    from person_type_a.engine import ClassifyTask
    from person_type_a.schema import QuestionSpec
    s = _chat_scorer()
    task = ClassifyTask("你是个图像分类器", "s", ("e",),
                        (QuestionSpec("ptype", "choice", "q？", ("a", "b")),))
    img = object()
    messages, slots = s.build_messages(task, image=img)
    assert messages[0]["role"] == "system"
    assert "图像分类器" in messages[0]["content"]
    user = messages[1]["content"]
    assert user[0] == {"type": "image", "image": img}         # 图像对象随块传入
    assert "答案 1：" in user[1]["text"] and "判定依据" not in user[1]["text"]
    assert [x.qid for x in slots] == ["ptype"]


def test_build_messages_without_image_has_no_image_block():
    from person_type_a.engine import ClassifyTask
    from person_type_a.schema import QuestionSpec
    s = _chat_scorer()
    task = ClassifyTask("y", "s", ("e",), (QuestionSpec("ptype", "choice", "q？", ("a", "b")),))
    messages, _ = s.build_messages(task)  # verify_anchors 用：无需真实图像
    assert all(c.get("type") != "image" for c in messages[1]["content"])


def test_verify_anchors_via_chat_template():
    from person_type_a.engine import ClassifyTask
    from person_type_a.schema import QuestionSpec
    s = _chat_scorer()
    task = ClassifyTask("y", "s", ("e",), (QuestionSpec("ptype", "choice", "q？", ("a", "b")),))
    assert s.verify_anchors(task) is True
