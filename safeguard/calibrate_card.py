"""safety / binary 分桶温度校准。

在拟合子集上网格搜索 NLL 最小的温度（复用 person_type_a/calibrator），
在互斥的 holdout 子集上报告 ECE 前后对比。拟合与评测数据严格分离——
在报告集上拟合温度等于在考卷上调参。

温度只重塑置信度（argmax 不变），因此 accuracy 应前后一致；
变化的是 ECE 与门控分段质量。

用法（仓库根目录）：
  CUDA_VISIBLE_DEVICES=1 ~/Projects/SafeGuardElderlyDomain/.venv/bin/python \
      -m safeguard.calibrate_card --model safeguard/output/card_rlcd/merged
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="答案卡分桶温度校准")
    ap.add_argument("--model", default="safeguard/output/card_rlcd/merged")
    ap.add_argument("--val", default="/data0/zhangpengyi/Projects/SafeGuardElderlyDomain/data/ElderDomain/hk_safeguard_val_subset1.jsonl")
    ap.add_argument("--fit-n", type=int, default=2000)
    ap.add_argument("--holdout-n", type=int, default=4000)
    ap.add_argument("--out", default="safeguard/output/card_rlcd/calibrator.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from person_type_a.calibrator import (ece, fit_temperature,
                                            save_calibrator)
    from .card import (ALL_CATEGORIES, SAFETY_LABELS, gold_slots,
                       read_card, temper_probs, wants_refusal)
    from .native import load_domain_template

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    domain_template = load_domain_template()

    samples = []
    with open(args.val) as f:
        for line in f:
            samples.append(json.loads(line))
    fit_set = samples[:args.fit_n]
    holdout = samples[args.fit_n:args.fit_n + args.holdout_n]
    print(f"fit={len(fit_set)} holdout={len(holdout)} (disjoint slices)")

    # ── 拟合：收集每槽 (log p 候选分数, gold 下标) ──
    s_fit: list[tuple[list[float], int]] = []
    b_fit: list[tuple[list[float], int]] = []
    for sample in fit_set:
        messages = sample["messages"]
        incl = wants_refusal(messages)
        gold = gold_slots(sample["assistant_label"], incl)
        if gold is None:
            continue
        r = read_card(model, tokenizer, messages, domain_template, incl)
        if r.error:
            continue
        s_fit.append(([math.log(max(r.safety_probs[l], 1e-9)) for l in SAFETY_LABELS],
                      SAFETY_LABELS.index(gold["safety"])))
        for cat in ALL_CATEGORIES:
            py = r.cat_yes[cat]
            b_fit.append(([math.log(max(py, 1e-9)), math.log(max(1 - py, 1e-9))],
                          1 - gold["cats"][cat]))
        if incl:
            py = r.refusal_probs["Yes"]
            b_fit.append(([math.log(max(py, 1e-9)), math.log(max(1 - py, 1e-9))],
                          0 if gold["refusal"] == "Yes" else 1))

    t_safety = fit_temperature(s_fit)
    t_binary = fit_temperature(b_fit)
    buckets = {"safety|3": t_safety, "binary|2": t_binary}
    save_calibrator(args.out, buckets)
    print(f"\nfitted: safety|3 T={t_safety}  binary|2 T={t_binary}  "
          f"(n={len(s_fit)}/{len(b_fit)}) → {args.out}")

    # ── holdout：ECE / 门控前后对比 ──
    s_confs0, s_hits0, s_confs1, s_hits1 = [], [], [], []
    b_confs0, b_hits0, b_confs1, b_hits1 = [], [], [], []
    s_correct = 0
    n_used = 0
    for sample in holdout:
        messages = sample["messages"]
        incl = wants_refusal(messages)
        gold = gold_slots(sample["assistant_label"], incl)
        if gold is None:
            continue
        r = read_card(model, tokenizer, messages, domain_template, incl)
        if r.error:
            continue
        n_used += 1

        sp0 = [r.safety_probs[l] for l in SAFETY_LABELS]
        sp1 = temper_probs(sp0, t_safety)
        ok = SAFETY_LABELS[sp0.index(max(sp0))] == gold["safety"]
        s_correct += ok
        s_confs0.append(max(sp0)); s_hits0.append(ok)
        s_confs1.append(max(sp1)); s_hits1.append(ok)

        for cat in ALL_CATEGORIES:
            py = r.cat_yes[cat]
            pred = py > 0.5
            g = bool(gold["cats"][cat])
            py1 = temper_probs([py, 1 - py], t_binary)[0]
            b_confs0.append(max(py, 1 - py)); b_hits0.append(pred == g)
            b_confs1.append(max(py1, 1 - py1)); b_hits1.append(pred == g)
        if incl:
            py = r.refusal_probs["Yes"]
            pred = py > 0.5
            g = gold["refusal"] == "Yes"
            py1 = temper_probs([py, 1 - py], t_binary)[0]
            b_confs0.append(max(py, 1 - py)); b_hits0.append(pred == g)
            b_confs1.append(max(py1, 1 - py1)); b_hits1.append(pred == g)

    def gate(confs, hits, thr=0.90):
        idx = [i for i, c in enumerate(confs) if c >= thr]
        cov = len(idx) / max(1, len(confs))
        acc = sum(hits[i] for i in idx) / max(1, len(idx))
        return cov, acc

    report = {
        "n_holdout": n_used,
        "temperatures": buckets,
        "safety": {"ece_raw": round(ece(s_confs0, s_hits0), 4),
                   "ece_cal": round(ece(s_confs1, s_hits1), 4),
                   "acc": round(s_correct / max(1, n_used), 4)},
        "binary": {"ece_raw": round(ece(b_confs0, b_hits0), 4),
                   "ece_cal": round(ece(b_confs1, b_hits1), 4),
                   "slots": len(b_confs0)},
        "gating_safety_auto90": {
            "raw": [round(v, 4) for v in gate(s_confs0, s_hits0)],
            "cal": [round(v, 4) for v in gate(s_confs1, s_hits1)]},
    }
    print()
    print("== holdout 校准前后（argmax 不变，accuracy 恒定）==")
    print(f"  safety ECE: {report['safety']['ece_raw']} → {report['safety']['ece_cal']}"
          f"   (acc {report['safety']['acc']})")
    print(f"  binary ECE: {report['binary']['ece_raw']} → {report['binary']['ece_cal']}"
          f"   ({report['binary']['slots']} slots)")
    cov0, acc0 = gate(s_confs0, s_hits0); cov1, acc1 = gate(s_confs1, s_hits1)
    print(f"  safety auto@0.90: 覆盖 {cov0:.3f}@{acc0:.4f} → 覆盖 {cov1:.3f}@{acc1:.4f}")

    out_json = Path(args.out).with_name("calibration_report.json")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {args.out}\n→ {out_json}")


if __name__ == "__main__":
    main()
