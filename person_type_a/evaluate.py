"""统一评测 CLI：字母槽（含 LoRA）、JSON 生成、零样本基线。

用法（仓库根目录）：
  # 字母槽评测（可挂载 LoRA）
  uv run --with torch --with transformers --with peft --with pillow --with torchvision \
      python -m person_type_a.evaluate \
      --model /path/to/gemma-4-E4B-it --mode letter --data-root ./data \
      [--lora output/cifar10_full/lora_adapter]

  # JSON 生成基线
  ... --mode json

  # 零样本（等价于 --mode letter --no-lora）
  ... --mode letter
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def eval_letter(args) -> dict:
    """字母槽评测：锚位提取 logits → 字母子集 softmax → accuracy + ECE。"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from .dataset import CIFAR10_CLASSES, CIFAR10LetterSlot, make_cifar10_scenario
    from .prompt import build_system_text, build_question_text_single, chat_messages
    from .schema import QuestionSpec
    from .transformers_scorer import find_subsequence

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
        print(f"LoRA loaded: {args.lora}")
    model.eval()

    scenario = make_cifar10_scenario()
    ds = CIFAR10LetterSlot(root=args.data_root, split=args.split,
                            augment_order=False, seed=args.seed)
    anchor_ids = tokenizer.encode("答案：", add_special_tokens=False)

    correct, total = 0, 0
    confs, hits = [], []
    t0 = time.perf_counter()

    for i in range(min(args.n_samples, len(ds))):
        img, label, meta = ds[i]
        q = scenario.questions[0]
        oq = QuestionSpec(
            qid=q.qid, kind=q.kind, instructions=q.instructions,
            options=tuple(meta["ordered"]),
            criteria=tuple("" for _ in meta["ordered"]))
        qtext = build_question_text_single(oq)
        msgs = chat_messages(
            build_system_text(scenario.system, scenario.scene, scenario.evidence),
            qtext, img)
        try:
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)

        inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]

        ids = inputs.input_ids[0].tolist()
        pos = find_subsequence(ids, anchor_ids)
        if pos < 0:
            continue

        letter_ids = []
        for letter in meta["letters"].values():
            full = tokenizer.encode("答案：" + letter, add_special_tokens=False)
            suffix = full[len(anchor_ids):]
            letter_ids.append(suffix[0] if suffix else tokenizer.encode(letter)[0])

        probs = torch.softmax(logits[pos].float()[letter_ids], dim=-1)
        pred_label = meta["ordered"][probs.argmax().item()]
        gold_label = CIFAR10_CLASSES[label]
        hit = pred_label == gold_label
        correct += hit
        total += 1
        confs.append(probs.max().item())
        hits.append(1.0 if hit else 0.0)

        if (i + 1) % args.report_every == 0:
            print(f"  {i+1}/{args.n_samples} acc={correct/total:.3f}")

    return _summarize(correct, total, confs, hits, t0, args, "letter")


def eval_json(args) -> dict:
    """JSON 生成评测：generate → parse → accuracy + parse rate。"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from .dataset import CIFAR10_CLASSES, CIFAR10LetterSlot
    from .json_baseline import parse_model_json

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()

    ds = CIFAR10LetterSlot(root=args.data_root, split=args.split,
                            augment_order=False, seed=args.seed)
    json_prompt = (
        "Classify the object in this image. "
        'Answer with a JSON object like {"answer": "class_name"} '
        f"where class_name is one of: {', '.join(CIFAR10_CLASSES)}. "
        "Output only the JSON.")

    correct, total, parse_ok = 0, 0, 0
    t0 = time.perf_counter()

    for i in range(min(args.n_samples, len(ds))):
        img, label, _ = ds[i]
        gold = CIFAR10_CLASSES[label]
        msgs = [
            {"role": "system", "content": "You are an image classifier."},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": json_prompt},
            ]},
        ]
        try:
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)

        inputs = processor(text=text, images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        gen = processor.tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        parsed = parse_model_json(gen)
        total += 1
        if parsed and "answer" in parsed:
            parse_ok += 1
            pred = str(parsed["answer"]).strip().lower()
            if pred == gold.lower():
                correct += 1

        if (i + 1) % args.report_every == 0:
            print(f"  {i+1}/{args.n_samples} acc={correct/total:.3f} "
                  f"parse={parse_ok/total:.3f}")

    dt = time.perf_counter() - t0
    result = {
        "mode": "json", "n_samples": total,
        "accuracy": correct / total, "parse_rate": parse_ok / total,
        "avg_latency_ms": round(dt / total * 1000), "eval_time_s": round(dt, 1),
    }
    return result


def _summarize(correct, total, confs, hits, t0, args, mode) -> dict:
    dt = time.perf_counter() - t0
    acc = correct / total if total else 0
    avg_conf = sum(confs) / len(confs) if confs else 0
    # ECE (10 bins)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = [j for j, c in enumerate(confs) if lo < c <= hi]
        if mask:
            bin_acc = sum(hits[j] for j in mask) / len(mask)
            bin_conf = sum(confs[j] for j in mask) / len(mask)
            ece += len(mask) / len(confs) * abs(bin_acc - bin_conf)
    return {
        "mode": mode, "n_samples": total,
        "accuracy": round(acc, 4), "avg_confidence": round(avg_conf, 4),
        "ece": round(ece, 4), "avg_latency_ms": round(dt / total * 1000),
        "eval_time_s": round(dt, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="统一评测 CLI")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["letter", "json"], default="letter")
    ap.add_argument("--lora", default=None, help="LoRA adapter 路径（letter 模式）")
    ap.add_argument("--split", default="val", choices=["train", "calibration", "val"])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--output", default=None, help="结果 JSON 路径")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    print(f"Mode: {args.mode}, LoRA: {args.lora or 'none'}, "
          f"split: {args.split}, n: {args.n_samples}")

    result = eval_letter(args) if args.mode == "letter" else eval_json(args)

    print(f"\n== Evaluation [{result['mode']}] ==")
    for k, v in result.items():
        if k != "mode":
            print(f"  {k:>16}: {v}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → {args.output}")


if __name__ == "__main__":
    main()
