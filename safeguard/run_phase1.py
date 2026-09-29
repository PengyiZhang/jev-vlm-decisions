"""Phase 1 driver：guard 模型零样本 jev 化三路线评测。

在同一 val 集上对比：
  (a) 自回归生成基线（复用 SafeGuardElderlyDomain 已有逐样本结果）
  (b) Route 1 原生锚定（guard 模板 + "Safety:"，safety 三分类）
  (c) Route 2a 每问独立标准 MCQ（锚在 prompt 末尾）
  (d) Route 2b 多问合一答题卡（单次前向读全部槽）

指标：safety 准确率/macro-F1/ECE、类别 per-cat F1/EM、refusal 准确率、
置信门控覆盖率、每样本延迟。运行示例见 safeguard/README.md。

本脚本需要 torch/transformers/sklearn，用 SafeGuardElderlyDomain 的 venv 运行
（该 venv 与训练环境一致）；纯逻辑部分无 GPU 依赖。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from person_type_a.readout import GATE_HI, GATE_LO, masked_softmax
from .mcq import build_2a_prompt, build_2b_prompt, read_positions
from .native import (
    SAFETY_LABELS,
    build_native_prompt,
    load_domain_template,
)
from .scenario import (
    ABSTAIN,
    ALL_CATEGORIES,
    build_questions,
    category_of_qid,
)
from .tokensurf import (
    check_continuation,
    letter_variants,
    lse_scores,
    safety_label_variants,
)

# 与 SafeGuardElderlyDomain 仓库一致的默认资产路径
DEFAULT_MODEL = Path(
    "/data0/zhangpengyi/Projects/SafeGuardElderlyDomain/outputs/merged_model_subset1_small"
)
DEFAULT_VAL = Path(
    "/data0/zhangpengyi/Projects/SafeGuardElderlyDomain/data/ElderDomain/hk_safeguard_val_subset1.jsonl"
)
DEFAULT_BASELINE = Path(
    "/data0/zhangpengyi/Projects/SafeGuardElderlyDomain/outputs/result_gen_elderly_0.6b_subset1_small.jsonl"
)


# ── 数据与标签解析 ───────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_guard_label(text: str) -> dict:
    """assistant_label / 生成输出 -> {safety, categories, refusal}。

    与 SafeGuardElderlyDomain/scripts/evaluate.py 的解析规则一致。
    """
    safety, cats, refusal = None, [], None
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("Safety:"):
            safety = line[len("Safety:"):].strip()
        elif line.startswith("Categories:"):
            s = line[len("Categories:"):].strip()
            cats = [] if s == "None" else [
                c.strip() for c in s.split(",") if c.strip() in ALL_CATEGORIES
            ]
        elif line.startswith("Refusal:"):
            refusal = line[len("Refusal:"):].strip()
    return {"safety": safety, "categories": cats, "refusal": refusal}


# ── 前向读取 ────────────────────────────────────────────────────────────────

def _forward_logits(model, input_ids, attention_mask) -> torch.Tensor:
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits.float()


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000.0


def run_route1(model, tok, samples, domain_template, label_variants, debug=False):
    """Route 1：每样本 1 次前向，末位 logits -> 三标签 LSE 分数 -> Softmax。"""
    records = []
    for i, s in enumerate(samples):
        prompt = build_native_prompt(tok, s["messages"], domain_template)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        logits, ms = _timed(
            lambda: _forward_logits(model, ids["input_ids"], ids["attention_mask"])
        )
        row = logits[0, -1].tolist()
        scores = lse_scores(row, [label_variants[l] for l in SAFETY_LABELS])
        probs = masked_softmax(scores)
        rec = {
            "i": i,
            "eval_type": s["eval_type"],
            "probs": dict(zip(SAFETY_LABELS, [round(p, 5) for p in probs])),
            "ms": round(ms, 2),
        }
        if debug and i < 2:
            top = torch.topk(logits[0, -1], 8)
            rec["_debug_top"] = [
                {"tok": tok.decode([t]), "p": round(v, 4)}
                for t, v in zip(top.indices.tolist(), torch.softmax(top.values, -1).tolist())
            ]
        records.append(rec)
    return records


def run_route2a(model, tok, samples, debug=False):
    """Route 2a：每问独立 MCQ，锚在 prompt 末尾；每样本一次批量前向（左 padding）。"""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    variants = {L: letter_variants(tok, L, "Answer: ") for L in "ABCD"}
    qcache: dict[str, tuple] = {}
    records = []
    for i, s in enumerate(samples):
        et = s["eval_type"]
        questions = qcache.setdefault(et, build_questions(et))
        prompts = [build_2a_prompt(s["messages"], questions, k) for k in range(len(questions))]
        enc = tok(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        logits, ms = _timed(
            lambda: _forward_logits(model, enc["input_ids"], enc["attention_mask"])
        )
        answers = {}
        debug_rows = []
        for k, q in enumerate(questions):
            scores = lse_scores(logits[k, -1].tolist(), [variants[L] for L in "ABCD"[: len(q.options)]])
            answers[q.qid] = dict(zip(q.options, [round(p, 5) for p in masked_softmax(scores)]))
            if debug and i < 1 and k in (0, len(questions) - 1):
                top = torch.topk(logits[k, -1], 8)
                debug_rows.append({"qid": q.qid, "top": [
                    {"tok": tok.decode([t]), "p": round(v, 4)}
                    for t, v in zip(top.indices.tolist(), torch.softmax(top.values, -1).tolist())
                ]})
        records.append({
            "i": i,
            "eval_type": et,
            "answers": answers,
            "ms": round(ms, 2),
            "n_forwards": len(prompts),
            **({"_debug": debug_rows} if debug_rows else {}),
        })
    return records


def run_route2b(model, tok, samples, fill="E", debug=False):
    """Route 2b：答题卡单次前向，offset 定位每槽读取位。fill 为空串时跑空槽消融。"""
    variants = {L: letter_variants(tok, L, "1. ") for L in "ABCD"}
    qcache: dict[str, tuple] = {}
    records = []
    for i, s in enumerate(samples):
        et = s["eval_type"]
        questions = qcache.setdefault(et, build_questions(et))
        prompt, dummy_pos = build_2b_prompt(s["messages"], questions, fill=fill)
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        mask = torch.ones_like(ids)
        logits, ms = _timed(lambda: _forward_logits(model, ids, mask))
        offsets = enc["offset_mapping"]
        poss = read_positions(offsets, dummy_pos)
        row_all = logits[0].tolist()
        answers = {}
        debug_rows = []
        for k, q in enumerate(questions):
            scores = lse_scores(row_all[poss[k]], [variants[L] for L in "ABCD"[: len(q.options)]])
            answers[q.qid] = dict(zip(q.options, [round(p, 5) for p in masked_softmax(scores)]))
            if debug and i < 1 and k in (0, len(questions) - 1):
                row = torch.tensor(row_all[poss[k]])
                top = torch.topk(row, 8)
                debug_rows.append({"qid": q.qid, "top": [
                    {"tok": tok.decode([t]), "p": round(v, 4)}
                    for t, v in zip(top.indices.tolist(), torch.softmax(top.values, -1).tolist())
                ]})
        records.append({
            "i": i,
            "eval_type": et,
            "answers": answers,
            "ms": round(ms, 2),
            "n_forwards": 1,
            **({"_debug": debug_rows} if debug_rows else {}),
        })
    return records


def run_gen_latency(model, tok, samples, domain_template, n):
    """生成基线延迟：与 evaluate.py 同配置（greedy、max_new_tokens=256、eos 早停）。"""
    times, ntoks = [], []
    for s in samples[:n]:
        prompt = tok.apply_chat_template(
            s["messages"], tokenize=False, chat_template=domain_template,
            add_generation_prompt=False)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        def _gen():
            with torch.inference_mode():
                return model.generate(
                    **ids, max_new_tokens=256, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )

        out, ms = _timed(_gen)
        times.append(ms)
        ntoks.append(int(out.shape[1] - ids["input_ids"].shape[1]))
    return {
        "n": len(times),
        "mean_ms": round(statistics.mean(times), 1),
        "p50_ms": round(statistics.median(times), 1),
        "mean_new_tokens": round(statistics.mean(ntoks), 1),
    }


# ── 指标 ────────────────────────────────────────────────────────────────────

def _pct(x): return round(100 * x, 2)


def ece_of(confs: list[float], hits: list[int], bins: int = 15) -> float:
    from person_type_a.calibrator import ece
    return round(ece(confs, [float(h) for h in hits], bins), 4)


def safety_block(preds, refs, probs_list):
    """preds/ref: 标签字符串；probs_list: {label: p}。"""
    from sklearn.metrics import accuracy_score, f1_score

    labels = ["Controversial", "Safe", "Unsafe"]
    acc = accuracy_score(refs, preds)
    f1m = f1_score(refs, preds, labels=labels, average="macro", zero_division=0)
    confs, hits = [], []
    gate = {"auto": [0, 0], "review": [0, 0], "human": [0, 0]}  # [n, correct]
    abstain_wins = 0
    for p, r, pr in zip(preds, refs, probs_list):
        top, conf = max(pr.items(), key=lambda kv: kv[1])
        confs.append(conf)
        hits.append(int(top == r))
        if top == ABSTAIN:  # 弃权胜出：语义上不可自动执行，恒走人工
            abstain_wins += 1
            band = "human"
        else:
            band = "auto" if conf >= GATE_HI else ("review" if conf >= GATE_LO else "human")
        gate[band][0] += 1
        gate[band][1] += int(top == r)
    out = {
        "n": len(preds),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1m, 4),
        "ece": ece_of(confs, hits),
        "abstain_wins": _pct(abstain_wins / len(preds)) if preds else 0.0,
    }
    for band, (n, c) in gate.items():
        out[f"gate_{band}"] = {"coverage": _pct(n / len(preds)) if preds else 0.0,
                               "accuracy": round(c / n, 4) if n else None}
    return out


def category_block(records, refs_cats):
    """每类二元判定（argmax == yes）+ 集合 EM。"""
    from sklearn.metrics import precision_recall_fscore_support

    per_cat, confs, hits, em = {}, [], [], 0
    for cat in ALL_CATEGORIES:
        qid = "cat_" + _slug(cat)
        golds, preds = [], []
        for rec, ref in zip(records, refs_cats):
            ans = rec["answers"][qid]
            top = max(ans.items(), key=lambda kv: kv[1])[0]
            gold = "yes" if cat in ref else "no"
            golds.append(gold == "yes")
            preds.append(top == "yes")
            confs.append(ans[top])
            hits.append(int(top == gold))
        p, r, f1, _ = precision_recall_fscore_support(
            golds, preds, average="binary", zero_division=0)
        per_cat[cat] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
    for rec, ref in zip(records, refs_cats):
        pred_set = {c for c in ALL_CATEGORIES
                    if max(rec["answers"]["cat_" + _slug(c)].items(), key=lambda kv: kv[1])[0] == "yes"}
        em += int(pred_set == set(ref))
    return {
        "per_category": per_cat,
        "f1_macro": round(statistics.mean(v["f1"] for v in per_cat.values()), 4),
        "exact_match": round(em / len(records), 4),
        "ece_top1_macro": ece_of(confs, hits),
    }


def _slug(cat: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_")


def refusal_block(records, refs_refusal):
    refs, preds, confs, hits = [], [], [], []
    for rec, ref in zip(records, refs_refusal):
        if ref is None:
            continue
        ans = rec["answers"]["refusal"]
        top = max(ans.items(), key=lambda kv: kv[1])[0]
        gold = "yes" if ref == "Yes" else "no"
        refs.append(gold == "yes")
        preds.append(top == "yes")
        confs.append(ans[top])
        hits.append(int(top == gold))
    acc = sum(int(p == r) for p, r in zip(preds, refs)) / len(refs) if refs else None
    return {"n": len(refs), "accuracy": round(acc, 4) if acc is not None else None,
            "ece_top1": ece_of(confs, hits) if refs else None}


def check_label_continuation(tok, prompt: str, label: str, variant_id: int) -> None:
    """原生锚点后接完整标签词，首个新增 token 必须是解析出的变体 id。

    Controversial 是多 token 词，只校验首 token（掩码 Softmax 只用首 token）。
    """
    base = tok.encode(prompt, add_special_tokens=False)
    full = tok.encode(prompt + " " + label, add_special_tokens=False)
    assert full[: len(base)] == base, f"标签 {label} 与锚点尾合并"
    assert full[len(base)] == variant_id, (
        f"标签 {label} 首续 token {full[len(base)]} != 变体 {variant_id}")


# ── 基线对齐与报告 ──────────────────────────────────────────────────────────

def load_baseline(path: Path, samples) -> dict | None:
    if not path or not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    preds = data.get("predictions", [])
    n = len(samples)
    if len(preds) < n:
        return None  # 覆盖不完整时放弃对齐
    aligned = []
    for i, s in enumerate(samples):
        if preds[i].get("messages") != s["messages"]:
            return None
        aligned.append(preds[i])
    refs = [parse_guard_label(s["assistant_label"]) for s in samples]
    gen_preds = [p["prediction"] for p in aligned]
    probs = []
    for g in gen_preds:
        pr = {"Controversial": 0.0, "Safe": 0.0, "Unsafe": 0.0}
        pr.update(_hard_conf(g["safety"]))
        probs.append(pr)
    safety = safety_block(
        [g["safety"] or "None" for g in gen_preds],
        [r["safety"] for r in refs],
        probs,
    )
    cats = {
        "f1_macro": round(statistics.mean(
            v["f1"] for v in data["metrics"]["categories"].values()), 4),
        "exact_match": round(data["metrics"]["category_exact_match"], 4),
        "note": "来自全量 val 的既有指标（硬判定无置信度，无 ECE/门控）",
    }
    return {"safety": safety, "categories": cats,
            "metrics_file": str(path)}


def _hard_conf(label: str | None) -> dict:
    """生成基线是硬判定，置信度置 1.0（argmax 语义），ECE 仅作参考。"""
    if label in ("Safe", "Unsafe", "Controversial"):
        return {label: 1.0}
    return {}


def write_report(out_dir: Path, summary: dict) -> None:
    lines = ["# Phase 1 零样本对比报告", ""]
    r1 = summary.get("route1", {})
    r2a = summary.get("route2a", {})
    r2b = summary.get("route2b", {})
    base = summary.get("baseline") or {}

    lines += ["## Safety 三分类", "",
              "| 系统 | 前向次数/样本 | acc | macro-F1 | ECE | auto 门控覆盖 | auto 段 acc | ms/样本 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, blk in (("自回归生成(基线)", base.get("safety")),
                      ("Route 1 原生锚定", r1.get("safety")),
                      ("Route 2a 每问MCQ", r2a.get("safety")),
                      ("Route 2b 答题卡", r2b.get("safety"))):
        if not blk:
            continue
        ms = blk.get("ms_mean")
        lines.append(
            f"| {name} | {blk.get('n_forwards', '—')} | {_pct(blk['accuracy'])} | "
            f"{_pct(blk['f1_macro'])} | {blk['ece']} | "
            f"{blk['gate_auto']['coverage']}% | {blk['gate_auto']['accuracy']} | "
            f"{ms if ms is not None else '—'} |")

    cat_rows = [("自回归生成(基线)", base.get("categories")),
                ("Route 2a 每问MCQ", r2a.get("categories")),
                ("Route 2b 答题卡", r2b.get("categories"))]
    lines += ["", "## 类别多标签（13 类二元）", "",
              "| 系统 | macro-F1 | 集合 EM | ECE(top1) |", "| --- | --- | --- | --- |"]
    for name, blk in cat_rows:
        if not blk:
            continue
        lines.append(f"| {name} | {_pct(blk['f1_macro'])} | "
                     f"{_pct(blk['exact_match'])} | {blk.get('ece_top1_macro', '—')} |")

    ref_rows = [("Route 2a", r2a.get("refusal")), ("Route 2b", r2b.get("refusal"))]
    lines += ["", "## Refusal（仅 assistant_response）", "", "| 系统 | acc | ECE |", "| --- | --- | --- |"]
    for name, blk in ref_rows:
        if not blk:
            continue
        lines.append(f"| {name} | {_pct(blk['accuracy'])} | {blk['ece_top1']} |")

    if summary.get("gen_latency"):
        lines += ["", "## 生成延迟（同卡实测）", "", "```json",
                  json.dumps(summary["gen_latency"], ensure_ascii=False, indent=2), "```"]
    lines += ["", "> 延迟 caveat：评测期间 GPU 有其他负载，绝对值偏高，相对关系可信。", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ── 主流程 ──────────────────────────────────────────────────────────────────

def sanity_checks(tok, samples, questions_uq, questions_ar, domain_template, label_variants):
    for l in "ABCDE":
        bare = tok.encode(l, add_special_tokens=False)
        assert len(bare) == 1, f"字母 {l} 非单 token: {bare}"
    for lab in SAFETY_LABELS:
        safety_label_variants(tok, lab)  # 可解析即通过
    # 原生锚点：完整标签词续写无合并，首 token 与解析变体一致
    p_native = build_native_prompt(tok, samples[0]["messages"], domain_template)
    for lab in SAFETY_LABELS:
        check_label_continuation(tok, p_native, lab, label_variants[lab][0])
    # 2a 末位：裸字母与空格字母两种续写形态都不与 prompt 尾合并
    p_2a = build_2a_prompt(samples[0]["messages"], questions_uq, 0)
    assert check_continuation(tok, p_2a, "A") and check_continuation(tok, p_2a, " A"), \
        "2a 末位续写合并"
    # 2b 锚定位可解析且递增
    prompt, dummy_pos = build_2b_prompt(samples[0]["messages"], questions_uq)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    poss = read_positions(enc["offset_mapping"], dummy_pos)
    assert all(b > a for a, b in zip(poss, poss[1:])), f"2b 锚位非递增: {poss}"
    assert poss[-1] < len(enc["input_ids"]) - 1, "2b 末槽读到序列尾之外"
    print(f"[sanity] 通过：字母单 token / 标签可解析 / 续写无合并 / 2b 锚位递增 "
          f"(2b prompt {len(enc['input_ids'])} tokens)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--val-file", default=str(DEFAULT_VAL))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                    help="生成基线 result JSON；不存在则跳过对齐")
    ap.add_argument("--domain-template", default=None)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "output" / "phase1"))
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 条（快速验证）")
    ap.add_argument("--routes", default="1,2a,2b")
    ap.add_argument("--2b-fill", choices=["dummy", "empty"], default="dummy",
                    help="答题卡槽位填充：哑字母（默认）或空槽消融")
    ap.add_argument("--gen-latency-n", type=int, default=150)
    ap.add_argument("--skip-gen-latency", action="store_true")
    ap.add_argument("--debug", action="store_true", help="前两样本打印读取位 top token")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    routes = {r.strip() for r in args.routes.split(",")}

    samples = load_jsonl(Path(args.val_file))
    if args.limit:
        samples = samples[: args.limit]
    refs = [parse_guard_label(s["assistant_label"]) for s in samples]
    print(f"[data] {len(samples)} 条 val（user_query "
          f"{sum(1 for s in samples if s['eval_type'] == 'user_query')} / "
          f"assistant_response {sum(1 for s in samples if s['eval_type'] == 'assistant_response')}）")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    domain_template = load_domain_template(args.domain_template)

    questions_uq = build_questions("user_query")
    questions_ar = build_questions("assistant_response")
    label_variants = {l: safety_label_variants(tok, l) for l in SAFETY_LABELS}
    sanity_checks(tok, samples, questions_uq, questions_ar, domain_template, label_variants)

    summary: dict = {"model": args.model, "n_samples": len(samples),
                     "val_file": str(args.val_file)}

    if "1" in routes:
        recs = run_route1(model, tok, samples, domain_template, label_variants, args.debug)
        _dump_jsonl(out_dir / "route1.jsonl", recs)
        preds = [max(r["probs"], key=r["probs"].get) for r in recs]
        blk = safety_block(preds, [r["safety"] for r in refs], [r["probs"] for r in recs])
        blk["n_forwards"] = 1
        blk["ms_mean"] = round(statistics.mean(r["ms"] for r in recs), 1)
        blk["ms_p50"] = round(statistics.median(r["ms"] for r in recs), 1)
        summary["route1"] = {"safety": blk}
        print(f"[route1] safety acc {blk['accuracy']} f1 {blk['f1_macro']} "
              f"ece {blk['ece']} ms {blk['ms_mean']}")
        if args.debug:
            _print_debug(recs[:2], "route1")

    for key, runner in (("2a", run_route2a), ("2b", run_route2b)):
        if key not in routes:
            continue
        if key == "2b":
            fill = "E" if getattr(args, "2b_fill") == "dummy" else ""
            recs = runner(model, tok, samples, fill=fill, debug=args.debug)
        else:
            recs = runner(model, tok, samples, debug=args.debug)
        _dump_jsonl(out_dir / f"route{key}.jsonl", recs)
        preds = [max(r["answers"]["safety"], key=r["answers"]["safety"].get) for r in recs]
        blk = safety_block(preds, [r["safety"] for r in refs],
                           [r["answers"]["safety"] for r in recs])
        blk["n_forwards"] = recs[0]["n_forwards"]
        blk["ms_mean"] = round(statistics.mean(r["ms"] for r in recs), 1)
        blk["ms_p50"] = round(statistics.median(r["ms"] for r in recs), 1)
        cats = category_block(recs, [r["categories"] for r in refs])
        refusal = refusal_block(recs, [r["refusal"] for r in refs])
        summary[f"route{key}"] = {
            "safety": blk, "categories": cats, "refusal": refusal,
            **({"fill": getattr(args, "2b_fill")} if key == "2b" else {}),
        }
        print(f"[route{key}] safety acc {blk['accuracy']} f1 {blk['f1_macro']} "
              f"ece {blk['ece']} | cat f1 {cats['f1_macro']} em {cats['exact_match']} "
              f"| refusal acc {refusal['accuracy']} | ms {blk['ms_mean']}")
        if args.debug:
            _print_debug(recs[:1], f"route{key}")

    if not args.skip_gen_latency and args.gen_latency_n > 0:
        lat = run_gen_latency(model, tok, samples, domain_template, args.gen_latency_n)
        summary["gen_latency"] = lat
        print(f"[gen] {lat}")

    summary["baseline"] = load_baseline(Path(args.baseline) if args.baseline else None, samples)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, summary)
    print(f"[done] {out_dir / 'report.md'}")


def _dump_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def _print_debug(recs, tag) -> None:
    for r in recs:
        if "_debug_top" in r:
            print(f"[debug:{tag}:i{r['i']}] top@anchor = "
                  + ", ".join(f"{d['tok']!r}:{d['p']}" for d in r["_debug_top"]))
        for d in r.get("_debug", []):
            print(f"[debug:{tag}:i{r['i']}:{d['qid']}] top@slot = "
                  + ", ".join(f"{t['tok']!r}:{t['p']}" for t in d["top"]))


if __name__ == "__main__":
    main()
