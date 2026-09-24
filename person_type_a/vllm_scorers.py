"""vLLM 双策略 Scorer。

策略一 VLLMPerQuestionScorer：每问独立请求（共享指令+图像前缀，靠 vLLM
多模态前缀缓存复用图像），首 token 的 top-K logprobs 即该槽分布。

策略二b VLLMPromptLogprobsScorer：单请求占位符布局 + prompt_logprobs，
图像必然只编码一次，不依赖缓存支持度；槽位分布条件于中性占位符常量。

两策略返回的行都是稀疏 dict（token_id → logprob）。logprob 本身是
全词表 log-softmax 值，readout.masked_softmax 在其上做温度缩放
（p^(1/T) 归一）与 logits 路径同为合法温度族；校准与推理须用同一 scorer。

vLLM 惰性导入：测试环境注入假引擎即可全链路验证；未装 vllm 时
sampling 参数退化为 SimpleNamespace。
"""
from __future__ import annotations

from types import SimpleNamespace

from .engine import ClassifyTask, SlotRow
from .prompt import build_layout, build_single_question_prompt
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
    def __init__(self, model_id: str, engine=None, topk: int = 20,
                 tokenizer=None, image_token: str = "<image>"):
        self.model_id = model_id
        self._engine = engine
        self._tokenizer = tokenizer
        self.topk = topk
        # 不同 VLM 的图像占位符不同（Qwen 系 <|image_pad|>、gemma-4 <|image|>），
        # 按所用模型传入；transformers_scorer.resolve_image_token 的规则同样适用。
        # GPU 选择不在参数里：vLLM 由 CUDA_VISIBLE_DEVICES 环境变量控制
        self.image_token = image_token

    def load(self, **llm_kwargs):
        """惰性建引擎。engine 已注入（测试）则跳过。"""
        if self._engine is not None:
            return
        from vllm import LLM

        self._engine = LLM(model=self.model_id, **llm_kwargs)
        if self._tokenizer is None:
            self._tokenizer = self._engine.get_tokenizer()


class VLLMPerQuestionScorer(_VLLMBase):
    """策略一：M 个独立请求。依赖前缀缓存复用图像，load 时自动开启
    enable_prefix_caching；若所用 vLLM 版本不支持多模态前缀缓存，
    图像 Prefill 会付 M 次，应改用策略二b。"""

    def load(self, **llm_kwargs):
        llm_kwargs.setdefault("enable_prefix_caching", True)
        super().load(**llm_kwargs)

    def slot_logits(self, task: ClassifyTask, image=None) -> dict[str, SlotRow]:
        params = _make_params(max_tokens=1, temperature=1.0, logprobs=self.topk)
        prompts = []
        for k, q in enumerate(task.questions, start=1):
            entry = {"prompt": build_single_question_prompt(task.system, task.scene, task.evidence, q, k,
                                                            image_token=self.image_token)}
            if image is not None:
                entry["multi_modal_data"] = {"image": image}
            prompts.append(entry)
        outputs = self._engine.generate(prompts, params)
        rows: dict[str, SlotRow] = {}
        for q, out in zip(task.questions, outputs):
            lp = out.outputs[0].logprobs[0]  # 首 token 位置的 top-K 分布
            rows[q.qid] = row_from_logprobs(lp)
        return rows


class VLLMPromptLogprobsScorer(_VLLMBase):
    """策略二b：单请求 + 占位符 + prompt_logprobs。"""

    PLACEHOLDER = "？"

    def slot_logits(self, task: ClassifyTask, image=None) -> dict[str, SlotRow]:
        assert self._tokenizer is not None, "load() 或构造时注入 tokenizer"
        params = _make_params(max_tokens=1, temperature=1.0, prompt_logprobs=self.topk)
        layout = build_layout(task.system, task.scene, task.evidence, task.questions,
                              placeholder=self.PLACEHOLDER, image_token=self.image_token)
        entry = {"prompt": layout.text}
        if image is not None:
            entry["multi_modal_data"] = {"image": image}
        out = self._engine.generate([entry], params)[0]

        ids = list(out.prompt_token_ids)
        prompt_logprobs = out.prompt_logprobs
        rows: dict[str, SlotRow] = {}
        for slot in layout.slots:
            anchor_ids = self._tokenizer.encode(slot.anchor + self.PLACEHOLDER,
                                                add_special_tokens=False)
            pos = find_subsequence(ids, anchor_ids)
            if pos < 0 or prompt_logprobs is None or prompt_logprobs[pos] is None:
                raise RuntimeError(
                    f"槽位 logprobs 缺失: {slot.anchor}（检查占位符定位 / topk 是否过小）"
                )
            rows[slot.qid] = row_from_logprobs(prompt_logprobs[pos])
        return rows
