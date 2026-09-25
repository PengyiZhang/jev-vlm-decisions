"""JSON 生成式基线：与传统"让 VLM 输出 JSON"路径对比结果与延迟。

与字母槽路径共用 chat_messages 组装（同模板、同图像注入），差异仅在
输出侧：自回归生成 JSON 文本并解析，vs 单前向读字母分布。

用法：
  uv run --with torch --with transformers --with pillow \
      python -m person_type_a.json_baseline \
      --scenario demos/person_type_a/scenarios/terminal-sim.json \
      --model ~/LLMs/Qwen3.8-27B --image crop.png --engine transformers
"""
from __future__ import annotations

import argparse
import json
import re
import time

from .prompt import build_system_text, chat_messages
from .schema import QuestionSpec, load_scenario


def parse_model_json(text: str) -> dict | None:
    """剥离代码围栏后解析 JSON 对象；失败返回 None。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def build_json_prompt(questions: tuple[QuestionSpec, ...]) -> str:
    lines = [
        "请逐题作答，输出一个 JSON 对象：键为各题的 qid，值为所选选项的原文；",
        "不要输出 JSON 以外的任何内容。",
    ]
    for q in questions:
        opts = " / ".join(q.effective_options)
        lines.append(f"{q.qid}：{q.instructions} 选项：{opts}")
    return "\n".join(lines)


def run_transformers(args, scenario, image):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    ).eval()

    messages = chat_messages(
        build_system_text(scenario.system, scenario.scene, scenario.evidence),
        build_json_prompt(scenario.questions), image,
    )
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(model.device)
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
    import torch

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
    dt = (time.perf_counter() - t0) * 1000
    gen = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True), dt, len(gen)


def run_vllm(args, scenario, image):
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model)
    tokenizer = llm.get_tokenizer()
    messages = chat_messages(
        build_system_text(scenario.system, scenario.scene, scenario.evidence),
        build_json_prompt(scenario.questions), image,
    )
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    t0 = time.perf_counter()
    out = llm.generate([{"prompt": text, "multi_modal_data": {"image": image}}], sp)[0]
    dt = (time.perf_counter() - t0) * 1000
    gen = out.outputs[0]
    return gen.text, dt, len(gen.token_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="JSON 生成式基线（对比字母槽单前向）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--engine", choices=["transformers", "vllm"], default="vllm")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from PIL import Image

    scenario = load_scenario(args.scenario)
    image = Image.open(args.image).convert("RGB")

    runner = run_transformers if args.engine == "transformers" else run_vllm
    text, dt_ms, n_tokens = runner(args, scenario, image)

    print(f"\n== JSON 基线 [{args.engine}]（{dt_ms:.0f} ms，{n_tokens} 个生成 token，"
          f"{len(scenario.questions)} 问）==")
    parsed = parse_model_json(text)
    if parsed is None:
        print("解析失败，原始输出：")
        print(text[:500])
    else:
        for q in scenario.questions:
            print(f"[{q.qid}] {parsed.get(q.qid, '<缺失>')}")


if __name__ == "__main__":
    main()
