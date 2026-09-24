"""真实 VLM 适配：chat template 组装 + 一次前向读所有答案槽。

torch/transformers 惰性导入，测试环境无依赖可跑逻辑（FakeChatProcessor 注入）。

已知边界（见 README）：
- 消息经 processor.apply_chat_template 组装，图像由 image content 注入，
  文本侧不再手工拼接图像占位符；
- 锚定位假设分词器对全文与锚子串的切分一致（BPE 边界效应下偶有偏移，
  load() 后用 verify_anchors(task) 自检）；
- 字母单 token 依赖词表，上线前跑 encoding.check_letters_single_token。
"""
from __future__ import annotations

from .engine import ClassifyTask, Scorer, SlotRow
from .prompt import build_question_text, build_system_text


def find_subsequence(hay: list[int], needle: list[int]) -> int:
    """返回 needle 在 hay 中最后一次出现的末元素下标；找不到返回 -1。"""
    n = len(needle)
    for i in range(len(hay) - n, -1, -1):
        if hay[i : i + n] == needle:
            return i + n - 1
    return -1


KNOWN_IMAGE_TOKENS = ("<|image_pad|>", "<|image|>", "<start_of_image>", "<image>")


def resolve_image_token(processor, tokenizer) -> str:
    """解析模型期望的图像占位符（vLLM 等裸文本路径使用）。

    优先读 processor.image_token（Qwen 系为 <|image_pad|>、gemma-4 为 <|image|>）；
    拿不到则按已知清单探测词表。Qwen 风格的 pad token 需包进
    <|vision_start|>/<|vision_end|> 标记（词表存在时）。
    """
    tok = getattr(processor, "image_token", None)
    if not tok:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        tok = next((t for t in KNOWN_IMAGE_TOKENS if t in vocab), "<image>")
    if tok == "<|image_pad|>":
        to_id = getattr(tokenizer, "convert_tokens_to_ids", None)
        if to_id and to_id("<|vision_start|>") is not None and to_id("<|vision_end|>") is not None:
            return "<|vision_start|><|image_pad|><|vision_end|>"
    return tok


def _ids_to_list(ids) -> list[int]:
    return ids.tolist() if hasattr(ids, "tolist") else list(ids)


class TransformersScorer:
    """单次前向，返回每个槽位锚末 token 处的 next-token Logits 行。"""

    def __init__(self, model_id: str, device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._tokenizer = None
        self.image_token = "<image>"  # 仅 vLLM 裸文本路径需要；本引擎走 chat template

    def _require_deps(self, raise_if_missing: bool = False) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError as e:
            if raise_if_missing:
                raise ImportError(
                    "需要 torch 与 transformers：uv run --with torch --with transformers ..."
                ) from e
            return False

    def load(self):
        self._require_deps(raise_if_missing=True)
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.image_token = resolve_image_token(self._processor, self._tokenizer)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype="auto", device_map=self.device
        ).eval()

    def build_messages(self, task: ClassifyTask, image=None) -> tuple[list[dict], tuple]:
        """场景 → chat 消息。图像对象随 image content 块传入，
        apply_chat_template(tokenize=True) 会加载它并产出 pixel_values。"""
        qtext, slots = build_question_text(task.questions)
        user_content = ([{"type": "image", "image": image}] if image is not None else []) + [
            {"type": "text", "text": qtext}
        ]
        messages = [
            {"role": "system", "content": build_system_text(task.system, task.scene, task.evidence)},
            {"role": "user", "content": user_content},
        ]
        return messages, slots

    def _tokenize(self, messages) -> dict:
        return self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )

    def verify_anchors(self, task: ClassifyTask) -> bool:
        """模板化分词后每个锚必须可定位且唯一。上线前必须通过。

        不带 image 块（真 processor 会加载块内图像对象，锚在文本区不受影响）。
        """
        assert self._processor is not None and self._tokenizer is not None, "先调用 load()"
        messages, slots = self.build_messages(task)
        ids = _ids_to_list(self._tokenize(messages)["input_ids"][0])
        for slot in slots:
            anchor_ids = self._tokenizer.encode(slot.anchor, add_special_tokens=False)
            if find_subsequence(ids, anchor_ids) < 0:
                return False
        return True

    def slot_logits(self, task: ClassifyTask, image=None) -> dict[str, SlotRow]:
        import torch

        assert self._model is not None, "先调用 load()"
        messages, slots = self.build_messages(task, image=image)
        inputs = self._tokenize(messages).to(self._model.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]  # [seq, vocab]
        ids = _ids_to_list(inputs["input_ids"][0])
        out: dict[str, SlotRow] = {}
        for slot in slots:
            anchor_ids = self._tokenizer.encode(slot.anchor, add_special_tokens=False)
            pos = find_subsequence(ids, anchor_ids)
            if pos < 0:
                raise RuntimeError(f"锚定位失败: {slot.anchor}（跑 verify_anchors 排查）")
            out[slot.qid] = logits[pos].float().tolist()  # 槽位 next-token Logits
        return out
