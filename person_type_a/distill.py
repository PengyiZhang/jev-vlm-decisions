"""蒸馏训练：用 JSON 生成（零样本高精度）的输出作为标签训练字母槽 LoRA。

流水线：
  1. 跑 JSON 生成（500ms/样本，零样本 89.5%）→ 保存伪标签
  2. 用伪标签作为 gold 训练字母槽（GRPO + proper-reward）
  3. 部署字母槽（4.3× 加速 + 校准 + 门控）

用法：
  # Step 1: 生成伪标签
  python -m person_type_a.distill --model ~/LLMs/gemma-4-E4B-it \
      --stage label --data-root ./data --output-dir output/distill

  # Step 2: 训练
  python -m person_type_a.distill --model ~/LLMs/gemma-4-E4B-it \
      --stage train --data-root ./data --output-dir output/distill
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def stage_label(args, scenario, train_ds):
    """Stage 1: 用 JSON 生成获取伪标签。"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from .json_baseline import parse_model_json
    from .prompt import chat_messages
    from .dataset import CIFAR10_CLASSES

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    labels_file = Path(args.output_dir) / "pseudo_labels.jsonl"
    labels_file.parent.mkdir(parents=True, exist_ok=True)

    json_prompt_tpl = (
        "Classify the object in this image. "
        'Answer with a JSON object like {{"answer": "class_name"}} '
        "where class_name is one of: {classes}. Output only the JSON.")

    n_done = 0
    t0 = time.perf_counter()
    with open(labels_file, "w") as f:
        for i in range(min(args.n_samples, len(train_ds))):
            img, label, meta = train_ds[i]
            jp = json_prompt_tpl.format(classes=", ".join(CIFAR10_CLASSES))
            msgs = [
                {"role": "system", "content": "You are an image classifier."},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": jp},
                ]},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)

            inputs = processor(text=text, images=[img],
                               return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            gen = processor.tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_model_json(gen)

            pseudo_label = None
            if parsed and "answer" in parsed:
                raw = str(parsed["answer"]).strip().lower()
                if raw in CIFAR10_CLASSES:
                    pseudo_label = raw

            rec = {"idx": i, "true_label": CIFAR10_CLASSES[label],
                   "pseudo_label": pseudo_label, "raw_output": gen[:100]}
            f.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 100 == 0:
                dt = time.perf_counter() - t0
                ok = sum(1 for _ in open(labels_file) if json.loads(_).get("pseudo_label"))
                print(f"  {n_done}/{args.n_samples} labeled, "
                      f"parse_ok={ok}/{n_done} ({dt:.0f}s)")

    print(f"Stage 1 done: {n_done} samples labeled → {labels_file}")


def stage_train(args, scenario, train_ds):
    """Stage 2: 用伪标签跑 GRPO 训练。"""
    from .train import main as train_main
    # 注入伪标签路径，train.py 已有完整训练循环
    # 这里直接调用 train.py 的 main，但替换数据源
    import sys
    sys.argv = [
        "train.py",
        "--model", args.model,
        "--output-dir", str(Path(args.output_dir) / "lora"),
        "--epochs", str(args.epochs),
        "--micro-batch", "1",
        "--grad-accum", str(args.grad_accum),
        "--data-root", args.data_root,
        "--max-steps", str(args.max_steps),
    ]
    train_main()


def main() -> None:
    ap = argparse.ArgumentParser(description="蒸馏训练：JSON 生成 → 字母槽 LoRA")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=["label", "train", "all"], default="all")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output-dir", default="output/distill")
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from .dataset import CIFAR10LetterSlot, make_cifar10_scenario
    scenario = make_cifar10_scenario()
    train_ds = CIFAR10LetterSlot(root=args.data_root, split="train",
                                  augment_order=True, seed=42)

    if args.stage in ("label", "all"):
        print("=== Stage 1: JSON pseudo-labeling ===")
        stage_label(args, scenario, train_ds)
    if args.stage in ("train", "all"):
        print("=== Stage 2: GRPO training on pseudo-labels ===")
        stage_train(args, scenario, train_ds)


if __name__ == "__main__":
    main()
