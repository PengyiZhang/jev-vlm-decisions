"""vLLM 双策略 Scorer（chat 模板路径）。

与 transformers 引擎共用 prompt.chat_messages + apply_chat_template 渲染，
三引擎输入一致，避免裸文本对 chat 模型构成 OOD 输入；图像占位符由模板
注入，经 multi_modal_data 传给 vLLM。

策略一 VLLMPerQuestionScorer：每问一组消息独立请求，add_generation_prompt
追加 assistant 开头后，生成位置（模型被训练作答的位置）的 top-K logprobs
即该槽分布；依赖 enable_prefix_caching 复用图像 Prefill。

策略二b VLLMPromptLogprobsScorer：单请求占位符布局 + prompt_logprobs，
图像必然只编码一次；槽位分布条件于中性占位符常量。

两策略返回稀疏 dict 行（token_id → logprob）。logprob 是全词表
log-softmax 值，readout.masked_softmax 在其上做温度缩放（p^(1/T) 归一）
与 logits 路径同为合法温度族；校准与推理须用同一 scorer。

vLLM 惰性导入：测试注入假引擎 + 假模板分词器即可全链路验证；未装
vllm 时 sampling 参数退化为 SimpleNamespace。
"""
from __future__ import annotations

from types import SimpleNamespace

from .engine import ClassifyTask, SlotRow
from .prompt import (
    build_question_text, build_question_text_single, build_system_text, chat_messages,
)
from .transformers_scorer import find_subsequence


def row_from_logprobs(logprob_dict: dict) -> dict[int, float]:
    """vLLM 的 {token_id: Logprob|float} → 裸 dict 行。"""
    return {tid: (lp.logprob if hasattr(lp, "logprob") else float(lp))
            for tid, lp in logprob_dict.items()}


def _make_params(**kw):
    """vLLM 可导入时用真 SamplingParams，否则退化命名空间（测试/无依赖环境）。"""
    try:
        from vllm import SamplingParams
        return SamplingParams(**kw)
    except ImportError:
        return SimpleNamespace(**kw)


class _VLLMBase:
    def __init__(self, model_id: str, engine=None, topk: int = 20, tokenizer=None):
        self.model_id = model_id
        self._engine = engine
        self._tokenizer = tokenizer
        self.topk = topk
        # GPU 选择不在参数里：vLLM 由 CUDA_VISIBLE_DEVICES 环境变量控制

    def load(self, **llm_kwargs):
        """惰性建引擎。engine 已注入（测试）则跳过。"""
        if self._engine is not None:
            return
        from vllm import LLM

        self._engine = LLM(model=self.model_id, **llm_kwargs)
        if self._tokenizer is None:
            self._tokenizer = self._engine.get_tokenizer()

    def _render(self, messages) -> str:
        assert self._tokenizer is not None, "load() 或构造时注入 tokenizer"
        try:
            # Qwen3.x 等模板默认在 assistant 头插入 <think>，生成位置落在思考块内，
            # 答案字母进不了 top-K；enable_thinking=False 让生成位置直接落在答案区
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )


class VLLMPerQuestionScorer(_VLLMBase):
    """策略一：M 组消息独立请求。load 自动开 enable_prefix_caching；
    若所用 vLLM 版本不支持多模态前缀缓存，图像 Prefill 会付 M 次，
    应改用策略二b。"""

    def load(self, **llm_kwargs):
        llm_kwargs.setdefault("enable_prefix_caching", True)
        super().load(**llm_kwargs)

    def slot_logits(self, task: ClassifyTask, image=None) -> dict[str, SlotRow]:
        params = _make_params(max_tokens=1, temperature=1.0, logprobs=self.topk)
        system_text = build_system_text(task.system, task.scene, task.evidence)
        prompts = []
        for k, q in enumerate(task.questions, start=1):
            messages = chat_messages(system_text, build_question_text_single(q, k), image)
            entry = {"prompt": self._render(messages)}
            if image is not None:
                entry["multi_modal_data"] = {"image": image}
            prompts.append(entry)
        outputs = self._engine.generate(prompts, params)
        rows: dict[str, SlotRow] = {}
        for q, out in zip(task.questions, outputs):
            lp = out.outputs[0].logprobs[0]  # 生成位置的 top-K 分布
            rows[q.qid] = row_from_logprobs(lp)
        return rows


class VLLMPromptLogprobsScorer(_VLLMBase):
    """策略二b：单请求 + 哑字母填充 + prompt_logprobs。

    槽位填超出候选集的哑字母（4 候选填 E）：in-context 锚定"答案=字母"
    格式；空槽会被强指令模型用 <|im_end|>/散文式作答挤掉字母（实测
    Qwen3.8 空槽位 im_end logprob≈0、字母掉出 top-K），中性符号"？"
    同样会被当成格式示例学舌。读哑字母位置的 prompt_logprobs，恰为
    P(·|锚为止) 的答案分布，哑字母本身不在候选集、被读取层排除。
    """

    def slot_logits(self, task: ClassifyTask, image=None) -> dict[str, SlotRow]:
        from .encoding import dummy_letter

        params = _make_params(max_tokens=1, temperature=1.0, prompt_logprobs=self.topk)
        qtext, slots = build_question_text(task.questions, fill_dummy=True)
        messages = chat_messages(
            build_system_text(task.system, task.scene, task.evidence), qtext, image
        )
        entry = {"prompt": self._render(messages)}
        if image is not None:
            entry["multi_modal_data"] = {"image": image}
        out = self._engine.generate([entry], params)[0]

        ids = list(out.prompt_token_ids)
        prompt_logprobs = out.prompt_logprobs
        rows: dict[str, SlotRow] = {}
        for slot, q in zip(slots, task.questions):
            fill = dummy_letter(len(q.effective_options))
            anchor_ids = self._tokenizer.encode(slot.anchor + fill, add_special_tokens=False)
            pos = find_subsequence(ids, anchor_ids)
            if pos < 0 or prompt_logprobs is None or prompt_logprobs[pos] is None:
                raise RuntimeError(
                    f"槽位 logprobs 缺失: {slot.anchor}（检查哑字母定位 / topk 是否过小）"
                )
            rows[slot.qid] = row_from_logprobs(prompt_logprobs[pos])
        return rows
