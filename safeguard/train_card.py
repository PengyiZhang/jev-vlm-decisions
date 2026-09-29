"""Native-Slot Answer Card RLCD 训练（GRPO + proper-reward，多槽）。

把 Qwen3Guard-Gen-0.6B 的变长三行输出重写为固定答案卡后训练：
- prompt = 对话渲染 + 空槽答案卡，与推理条件化完全一致
- 每个槽位独立计算 proper reward（GRPO 组内归约 + CE 引导），损失为槽间平均
- safety 槽 3 候选 / 类别与 refusal 槽 2 候选（Yes 恒为索引 0）

用法（仓库根目录）：
  CUDA_VISIBLE_DEVICES=1 ~/Projects/SafeGuardElderlyDomain/.venv/bin/python \
      -m safeguard.train_card \
      --output-dir safeguard/output/card_rlcd
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="答案卡 RLCD 训练（guard 全槽覆盖）")
    ap.add_argument("--model", default="/data0/zhangpengyi/LLMs/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--train",
                    default="/data0/zhangpengyi/Projects/SafeGuardElderlyDomain/data/ElderDomain/hk_safeguard_train_subset1.jsonl")
    ap.add_argument("--output-dir", default="safeguard/output/card_rlcd")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma-max", type=float, default=0.2)
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--w-sph", type=float, default=0.75)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--pure-ce", action="store_true", help="对照实验：纯 CE，无 GRPO")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048, help="超长样本跳过（截断会切掉答案卡）")
    ap.add_argument("--max-steps", type=int, default=0, help="0=all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from person_type_a.grpo_trainer import GRPOLoss, sigma_schedule
    from .card import (ALL_CATEGORIES, SAFETY_LABELS, gold_slots,
                       locate_slots, render_card_prompt, safety_token_ids,
                       wants_refusal, yes_no_token_ids)
    from .native import load_domain_template

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    samples = []
    with open(args.train) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"train samples: {len(samples)}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    loss_fn = GRPOLoss(G=args.G, sigma=args.sigma_max,
                       w_sph=args.w_sph, lambda_ce=args.lambda_ce)

    domain_template = load_domain_template()
    s_ids = safety_token_ids(tokenizer)
    y_id, n_id = yes_no_token_ids(tokenizer)

    total_steps = len(samples) * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    log_file = out_dir / "train_log.jsonl"
    step = skipped = 0
    t_start = time.time()
    print(f"Total steps: {total_steps}")

    for epoch in range(args.epochs):
        indices = torch.randperm(len(samples)).tolist()
        for idx in indices:
            if step >= total_steps:
                break

            model.train()
            sample = samples[idx]
            messages = sample["messages"]
            incl = wants_refusal(messages)
            gold = gold_slots(sample["assistant_label"], incl)
            if gold is None:
                skipped += 1
                step += 1
                continue

            prompt = render_card_prompt(tokenizer, messages, domain_template, incl)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            if inputs.input_ids.shape[1] > args.max_len:
                skipped += 1
                step += 1
                continue

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits

            ids = inputs.input_ids[0].tolist()
            try:
                pos = locate_slots(tokenizer, ids, incl)
            except ValueError:
                skipped += 1
                step += 1
                continue

            # 每槽 (mu, gold one-hot)，损失为槽间平均
            slot_losses = []

            mu = logits[0, pos["safety"]].float()[s_ids].unsqueeze(0)
            g = torch.zeros(1, len(s_ids), device=model.device)
            g[0, SAFETY_LABELS.index(gold["safety"])] = 1.0
            slot_losses.append(("safety", mu, g))

            for cat in ALL_CATEGORIES:
                mu = logits[0, pos[f"cat::{cat}"]].float()[[y_id, n_id]].unsqueeze(0)
                g = torch.zeros(1, 2, device=model.device)
                g[0, 1 - gold["cats"][cat]] = 1.0  # Yes=0 / No=1
                slot_losses.append((f"cat", mu, g))

            if incl:
                mu = logits[0, pos["refusal"]].float()[[y_id, n_id]].unsqueeze(0)
                g = torch.zeros(1, 2, device=model.device)
                g[0, 0 if gold["refusal"] == "Yes" else 1] = 1.0
                slot_losses.append(("refusal", mu, g))

            sigma = 0.0
            if args.pure_ce:
                loss = sum(F.cross_entropy(mu, g)
                           for _, mu, g in slot_losses) / len(slot_losses)
            else:
                sigma = sigma_schedule(step, total_steps,
                                        args.sigma_max, args.sigma_min)
                loss_fn.sigma = sigma
                loss = sum(loss_fn(mu, g)
                           for _, mu, g in slot_losses) / len(slot_losses)
            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 50 == 0:
                with torch.no_grad():
                    safety_loss = float(F.cross_entropy(
                        slot_losses[0][1], slot_losses[0][2]))
                rec = {"step": step, "total": total_steps,
                       "loss": round(float(loss.item()) * args.grad_accum, 4),
                       "safety_ce": round(safety_loss, 4),
                       "sigma": round(sigma, 4), "skipped": skipped,
                       "elapsed_s": round(time.time() - t_start, 1)}
                print(f"step {step}/{total_steps} loss={rec['loss']} "
                      f"safety_ce={rec['safety_ce']} σ={rec['sigma']} "
                      f"skip={skipped} t={rec['elapsed_s']}s")
                with open(log_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        if step >= total_steps:
            break

    print(f"Saving LoRA adapter... (skipped {skipped})")
    model.save_pretrained(str(out_dir / "lora_adapter"))
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
