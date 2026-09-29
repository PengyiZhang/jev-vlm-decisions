"""Phase 2 链式读取评测：Safety + Categories + Refusal 全链覆盖。

用法（仓库根目录）：
  CUDA_VISIBLE_DEVICES=1 ~/Projects/SafeGuardElderlyDomain/.venv/bin/python \
      -m safeguard.run_phase2 --n 200 --out-dir safeguard/output/phase2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .chain import chain_read
from .native import load_domain_template
from .scenario import ALL_CATEGORIES


def parse_generated(text: str) -> dict:
    """解析生成式三行输出为标准 dict。"""
    result = {"safety": "", "categories": [], "refusal": ""}
    for line in text.strip().split("\n"):
        if line.startswith("Safety:"):
            result["safety"] = line[len("Safety:"):].strip()
        elif line.startswith("Categories:"):
            cats = line[len("Categories:"):].strip()
            result["categories"] = [c.strip() for c in cats.split(",") if c.strip()]
        elif line.startswith("Refusal:"):
            result["refusal"] = line[len("Refusal:"):].strip()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 链式读取评测")
    ap.add_argument("--model", default="/data0/zhangpengyi/LLMs/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--val", default="/data0/zhangpengyi/Projects/SafeGuardElderlyDomain/data/ElderDomain/hk_safeguard_val_subset1.jsonl")
    ap.add_argument("--n", type=int, default=200, help="评测样本数")
    ap.add_argument("--out-dir", default="safeguard/output/phase2")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    domain_template = load_domain_template()

    # 加载 val 集
    samples = []
    with open(args.val) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.n:
                break
    print(f"Loaded {len(samples)} samples")

    correct_safety = 0
    correct_refusal = 0
    cat_tp = 0; cat_fp = 0; cat_fn = 0
    total_fwd = 0
    total_ms = 0.0
    details = []

    for i, sample in enumerate(samples):
        messages = sample["messages"]
        gold = parse_generated(sample["assistant_label"])

        t0 = time.perf_counter()
        result = chain_read(model, tokenizer, messages, domain_template)
        dt = (time.perf_counter() - t0) * 1000
        total_ms += dt
        total_fwd += result.forward_count

        # 评分
        is_refusal_sample = bool(gold.get("refusal"))
        if is_refusal_sample:
            if result.refusal_label == gold["refusal"]:
                correct_refusal += 1

        if result.safety_label == gold["safety"]:
            correct_safety += 1

        gold_cats = set(gold["categories"])
        pred_cats = set(result.categories)
        cat_tp += len(gold_cats & pred_cats)
        cat_fp += len(pred_cats - gold_cats)
        cat_fn += len(gold_cats - pred_cats)

        if (i + 1) % 50 == 0:
            n = i + 1
            print(f"  {n}/{len(samples)} safety={correct_safety/n:.3f} "
                  f"cat_F1={2*cat_tp/max(1,2*cat_tp+cat_fp+cat_fn):.3f} "
                  f"fwd={total_fwd/n:.1f} {total_ms/n:.0f}ms")

        details.append({
            "idx": i,
            "gold_safety": gold["safety"],
            "pred_safety": result.safety_label,
            "gold_cats": sorted(gold_cats),
            "pred_cats": sorted(pred_cats),
            "gold_refusal": gold.get("refusal", ""),
            "pred_refusal": result.refusal_label,
            "forwards": result.forward_count,
            "ms": round(dt, 1),
        })

    n = len(samples)
    n_refusal = sum(1 for d in details if d["gold_refusal"])
    cat_p = cat_tp / max(1, cat_tp + cat_fp)
    cat_r = cat_tp / max(1, cat_tp + cat_fn)
    cat_f1 = 2 * cat_p * cat_r / max(1e-9, cat_p + cat_r)

    print()
    print("== Phase 2 链式读取结果 ==")
    print(f"  samples:       {n}")
    print(f"  safety acc:    {correct_safety/n:.4f}")
    print(f"  cat P/R/F1:    {cat_p:.3f} / {cat_r:.3f} / {cat_f1:.3f}")
    print(f"  refusal acc:   {correct_refusal}/{n_refusal} = {correct_refusal/max(1,n_refusal):.4f}")
    print(f"  avg forwards:  {total_fwd/n:.1f}")
    print(f"  avg latency:   {total_ms/n:.0f} ms")

    report = {
        "n": n,
        "safety_acc": round(correct_safety / n, 4),
        "category_precision": round(cat_p, 4),
        "category_recall": round(cat_r, 4),
        "category_f1": round(cat_f1, 4),
        "refusal_acc": round(correct_refusal / max(1, n_refusal), 4),
        "refusal_n": n_refusal,
        "avg_forwards": round(total_fwd / n, 1),
        "avg_latency_ms": round(total_ms / n, 0),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "details.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n→ {out_dir}/summary.json")


if __name__ == "__main__":
    main()
