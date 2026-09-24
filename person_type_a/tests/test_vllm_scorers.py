import sys
from types import SimpleNamespace

from person_type_a.engine import CharTokenizer, ClassifyTask
from person_type_a.prompt import build_question_text, build_system_text, chat_messages
from person_type_a.schema import QuestionSpec
from person_type_a.transformers_scorer import find_subsequence
from person_type_a.vllm_scorers import (
    VLLMPerQuestionScorer, VLLMPromptLogprobsScorer, row_from_logprobs,
)

ASSISTANT_TAIL = "<assistant>"


class FakeLogprob:
    """仿真 vLLM 的 Logprob 对象（.logprob 属性）。"""
    def __init__(self, logprob: float):
        self.logprob = logprob


class ChatTemplateTokenizer(CharTokenizer):
    """字符级假分词器 + chat 模板渲染仿真。"""
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
        if kw.get("add_generation_prompt"):
            text += ASSISTANT_TAIL
        return text


def task():
    return ClassifyTask("y", "s", ("e",), (
        QuestionSpec("ptype", "choice", "人员类型？", ("a", "b")),
        QuestionSpec("vest", "binary", "是否穿反光背心？"),
    ))


def test_row_from_logprobs_accepts_object_and_float():
    row = row_from_logprobs({65: FakeLogprob(-0.2), 66: -1.7})
    assert row == {65: -0.2, 66: -1.7}


def test_per_question_renders_chat_template_per_question():
    seen = []

    class Engine:
        def generate(self, prompts, params):
            seen.append((list(prompts), params))
            row = {65: FakeLogprob(-0.1), 66: FakeLogprob(-2.0)}
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[row])])
                    for _ in prompts]

    s = VLLMPerQuestionScorer("m", engine=Engine(), tokenizer=ChatTemplateTokenizer(), topk=5)
    rows = s.slot_logits(task())

    prompts, params = seen[0]
    assert len(prompts) == 2                                   # 每问一组消息
    assert all(p["prompt"].startswith("<s>") and p["prompt"].endswith(ASSISTANT_TAIL)
               for p in prompts)                                # 模板渲染 + assistant 开头
    assert all("答案：" in p["prompt"] for p in prompts)
    assert "是否穿反光背心" not in prompts[0]["prompt"]          # 问题互不可见
    assert "y" in prompts[0]["prompt"]                          # 系统区在模板内
    assert params.max_tokens == 1 and params.logprobs == 5
    assert rows["ptype"] == {65: -0.1, 66: -2.0}


def test_per_question_attaches_image():
    seen = []

    class Engine:
        def generate(self, prompts, params):
            seen.extend(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{}])]) for _ in prompts]

    s = VLLMPerQuestionScorer("m", engine=Engine(), tokenizer=ChatTemplateTokenizer(), topk=5)
    s.slot_logits(task(), image=object())
    assert all("multi_modal_data" in p for p in seen)
    assert all("[IMG]" in p["prompt"] for p in seen)            # image 块被模板渲染


def test_prompt_logprobs_reads_placeholder_positions():
    tok = ChatTemplateTokenizer()
    t = task()
    qtext, slots = build_question_text(t.questions, placeholder=VLLMPromptLogprobsScorer.PLACEHOLDER)
    text = tok.apply_chat_template(
        chat_messages(build_system_text(t.system, t.scene, t.evidence), qtext),
        add_generation_prompt=True,
    )
    ids = [ord(c) for c in text]
    lp = [None] * len(ids)
    expected = {}
    for slot in slots:
        anchor = slot.anchor + VLLMPromptLogprobsScorer.PLACEHOLDER
        pos = find_subsequence(ids, [ord(c) for c in anchor])
        row = {65: FakeLogprob(-0.1), 66: FakeLogprob(-2.0)}
        lp[pos] = row
        expected[slot.qid] = {65: -0.1, 66: -2.0}

    class Engine:
        def generate(self, prompts, params):
            assert len(prompts) == 1                            # 单请求
            assert getattr(params, "prompt_logprobs") == 5
            return [SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=lp,
                                    outputs=[SimpleNamespace(logprobs=[None])])]

    s = VLLMPromptLogprobsScorer("m", engine=Engine(), tokenizer=tok, topk=5)
    assert s.slot_logits(t) == expected


def test_prompt_logprobs_raises_on_missing_position():
    class Engine:
        def generate(self, prompts, params):
            # prompt_logprobs 全 None：模拟 topk 太小或引擎未返回
            return [SimpleNamespace(prompt_token_ids=[ord(c) for c in prompts[0]["prompt"]],
                                    prompt_logprobs=[None] * len(prompts[0]["prompt"]),
                                    outputs=[SimpleNamespace(logprobs=[None])])]

    s = VLLMPromptLogprobsScorer("m", engine=Engine(), tokenizer=ChatTemplateTokenizer(), topk=5)
    try:
        s.slot_logits(task())
        raise AssertionError("should raise")
    except RuntimeError as e:
        assert "槽位" in str(e)


def test_load_never_passes_device_and_perq_enables_prefix_caching(monkeypatch):
    recorded = {}

    class FakeLLM:
        def __init__(self, **kw):
            recorded.update(kw)
        def get_tokenizer(self):
            return CharTokenizer()

    fake = SimpleNamespace(LLM=FakeLLM)
    monkeypatch.setitem(sys.modules, "vllm", fake)

    s1 = VLLMPerQuestionScorer("m1", topk=5)
    s1.load()
    assert recorded.get("model") == "m1"
    assert "device" not in recorded            # vLLM 无 device 参数，GPU 由 CUDA_VISIBLE_DEVICES 控制
    assert recorded.get("enable_prefix_caching") is True

    s2 = VLLMPromptLogprobsScorer("m2", tokenizer=CharTokenizer(), topk=5)
    s2.load()
    assert s2._engine is not None and "device" not in recorded
