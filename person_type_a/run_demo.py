"""路线 A 现场演示 CLI。

用法（仓库根目录，模块方式运行以启用包内相对导入）：
  uv run --with torch --with transformers --with pillow python -m person_type_a.run_demo \
      --scenario demos/person_type_a/scenarios/terminal.json \
      --model Qwen/Qwen3.8-27B-Instruct --image crop1.jpg [--image crop2.jpg ...] \
      [--engine transformers|vllm-perq|vllm-plogprob] [--calibrator calibrator.json]
"""
from __future__ import annotations

import argparse
import time

from .calibrator import load_calibrator
from .classify import classify
from .encoding import check_letters_single_token
from .schema import load_scenario, validate
from .transformers_scorer import TransformersScorer


def make_scorer(args):
    if args.engine == "transformers":
        from .transformers_scorer import TransformersScorer as T
        return T(args.model, device=args.device)
    if args.engine == "vllm-perq":
        from .vllm_scorers import VLLMPerQuestionScorer
        return VLLMPerQuestionScorer(args.model, topk=args.topk)
    if args.engine == "vllm-plogprob":
        from .vllm_scorers import VLLMPromptLogprobsScorer
        return VLLMPromptLogprobsScorer(args.model, topk=args.topk)
    raise SystemExit(f"未知引擎: {args.engine}")


def main() -> None:
    ap = argparse.ArgumentParser(description="路线 A：字母槽单次前向人员类型判定")
    ap.add_argument("--scenario", required=True, help="场景 JSON 路径")
    ap.add_argument("--model", required=True, help="VLM 模型 id（如 Qwen/Qwen3.8-27B-Instruct）")
    ap.add_argument("--image", action="append", required=True, help="crop 图路径，可多次")
    ap.add_argument("--engine", default="transformers",
                    choices=["transformers", "vllm-perq", "vllm-plogprob"],
                    help="transformers=单前向直读；vllm-perq=策略一每问独立请求；vllm-plogprob=策略二b占位符")
    ap.add_argument("--topk", type=int, default=20, help="vLLM logprobs 的 K（须 ≥ 候选字母数）")
    ap.add_argument("--calibrator", default=None, help="calibrator.json 路径（可选）")
    ap.add_argument("--device", default="auto",
                    help="仅 transformers 引擎生效；vLLM 引擎用 CUDA_VISIBLE_DEVICES 选卡")
    args = ap.parse_args()

    scenario = load_scenario(args.scenario)
    validate(scenario)
    max_k = max(len(q.effective_options) for q in scenario.questions)
    if args.engine.startswith("vllm") and args.topk < max_k:
        raise SystemExit(f"--topk {args.topk} 小于最大候选数 {max_k}，字母覆盖不全")

    temperatures = load_calibrator(args.calibrator) if args.calibrator else {}

    scorer = make_scorer(args)
    scorer.load()
    from transformers import AutoTokenizer

    bad = check_letters_single_token(AutoTokenizer.from_pretrained(args.model))
    if bad:
        raise SystemExit(f"字母非单 token，无法走路线 A：{bad}")
    if args.engine == "transformers":
        from .engine import ClassifyTask

        task = ClassifyTask(scenario.system, scenario.scene, scenario.evidence, scenario.questions)
        if not scorer.verify_anchors(task):
            raise SystemExit("锚定位自检失败：BPE 边界偏移，检查槽锚文本")

    from PIL import Image

    for path in args.image:
        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        results = classify(scorer, scorer._tokenizer, scenario.system, scenario.scene, scenario.evidence,
                           scenario.questions, image=image, temperatures=temperatures)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n== {path}（{dt:.0f} ms，单次前向 {len(scenario.questions)} 问）==")
        for qid, r in results.items():
            dist = "  ".join(f"{k} ({v:.2f})" for k, v in r["distribution"].items() if v >= 0.01)
            print(f"[{qid}] {r['top_label']} ({r['top_prob']:.2f})  gate={r['gate']}  "
                  f"abstain={r['abstain_mass']:.2f}")
            print(f"    {dist}")


if __name__ == "__main__":
    main()
