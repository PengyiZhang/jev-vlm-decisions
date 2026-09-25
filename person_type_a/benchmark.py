"""预热 + 多次重复推理的速度基准。

消除首轮开销（权重加载、CUDA graph 捕获、JIT 编译、前缀缓存冷启动），
测稳态延迟：先 warmup 轮，再计时 runs 轮，报 min/mean/median/p95/max。

用法（仓库根目录）：
  uv run --with torch --with transformers --with pillow \
      python -m person_type_a.benchmark \
      --scenario demos/person_type_a/scenarios/terminal-sim.json \
      --model ~/LLMs/Qwen3.8-27B --image crop.png \
      --engine vllm-perq --warmup 2 --runs 10
"""
from __future__ import annotations

import argparse
import statistics
import time


def summarize(times_ms: list[float]) -> dict[str, float]:
    """稳态统计：min/mean/median/p95/max（ms）。"""
    xs = sorted(times_ms)

    def pct(p: float) -> float:
        k = (len(xs) - 1) * p / 100.0
        f, c = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[f] + (xs[c] - xs[f]) * (k - f)

    return {
        "min": xs[0],
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p95": pct(95),
        "max": xs[-1],
    }


def _make_letter_scorer(args):
    if args.engine == "transformers":
        from .transformers_scorer import TransformersScorer
        return TransformersScorer(args.model, device=args.device)
    if args.engine == "vllm-perq":
        from .vllm_scorers import VLLMPerQuestionScorer
        return VLLMPerQuestionScorer(args.model, topk=args.topk)
    from .vllm_scorers import VLLMPromptLogprobsScorer
    return VLLMPromptLogprobsScorer(args.model, topk=args.topk)


def bench_letter_slot(args, scenario, image) -> list[float]:
    from .classify import classify

    scorer = _make_letter_scorer(args)
    scorer.load()
    for _ in range(args.warmup):
        classify(scorer, scorer._tokenizer, scenario.system, scenario.scene,
                 scenario.evidence, scenario.questions, image=image)
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        classify(scorer, scorer._tokenizer, scenario.system, scenario.scene,
                 scenario.evidence, scenario.questions, image=image)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _json_messages(scenario):
    from .json_baseline import build_json_prompt
    from .prompt import build_system_text, chat_messages
    return chat_messages(
        build_system_text(scenario.system, scenario.scene, scenario.evidence),
        build_json_prompt(scenario.questions),
    )


def bench_json_transformers(args, scenario, image) -> tuple[list[float], int]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device).eval()
    messages = _json_messages(scenario)
    messages[1]["content"].insert(0, {"type": "image", "image": image})
    try:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    n_in = inputs["input_ids"].shape[1]

    def once():
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        return (time.perf_counter() - t0) * 1000, int((out.shape[1] - n_in))

    for _ in range(args.warmup):
        once()
    pairs = [once() for _ in range(args.runs)]
    return [t for t, _ in pairs], int(statistics.fmean(n for _, n in pairs))


def bench_json_vllm(args, scenario, image) -> tuple[list[float], int]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model)
    tokenizer = llm.get_tokenizer()
    messages = _json_messages(scenario)
    messages[1]["content"].insert(0, {"type": "image", "image": image})
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    entry = {"prompt": text, "multi_modal_data": {"image": image}}

    def once():
        t0 = time.perf_counter()
        out = llm.generate([entry], sp)[0]
        return (time.perf_counter() - t0) * 1000, len(out.outputs[0].token_ids)

    for _ in range(args.warmup):
        once()
    pairs = [once() for _ in range(args.runs)]
    return [t for t, _ in pairs], int(statistics.fmean(n for _, n in pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description="预热 + 重复推理速度基准")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--engine", required=True,
                    choices=["transformers", "vllm-perq", "vllm-plogprob",
                             "json-transformers", "json-vllm"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from PIL import Image

    from .schema import load_scenario

    scenario = load_scenario(args.scenario)
    image = Image.open(args.image).convert("RGB")

    if args.engine.startswith("json"):
        times, n_tokens = (bench_json_transformers if args.engine == "json-transformers"
                           else bench_json_vllm)(args, scenario, image)
        s = summarize(times)
        print(f"\n== bench [{args.engine}] warmup={args.warmup} runs={args.runs} "
              f"~{n_tokens} gen tokens/run ==")
    else:
        times = bench_letter_slot(args, scenario, image)
        s = summarize(times)
        print(f"\n== bench [{args.engine}] warmup={args.warmup} runs={args.runs} "
              f"{len(scenario.questions)} questions ==")
    for k in ("min", "mean", "median", "p95", "max"):
        print(f"{k:>7}: {s[k]:8.1f} ms")


if __name__ == "__main__":
    main()
