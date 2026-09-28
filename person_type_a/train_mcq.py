"""标准 MCQ 格式 RLCD 训练（GRPO + proper-reward）。

与 train.py 的区别：prompt 使用标准 MCQ 格式（`A. option\nAnswer:`），
起跑线为零样本 86.5%（gemma-4-E4B），GRPO 在此基础上精细化校准。

用法（仓库根目录）：
  uv run --with torch --with transformers --with peft --with pillow --with torchvision \
      python -m person_type_a.train_mcq \
      --model /path/to/gemma-4-E4B-it --output-dir output/cifar10_mcq_rlcd
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_mcq_prompt(instructions: str, options: list[str]) -> str:
    """标准 MCQ prompt：`A. option` 格式，`Answer:` 锚。"""
    lines = [f"{instructions}", ""]
    for j, opt in enumerate(options):
        letter = chr(ord("A") + j)
        lines.append(f"{letter}. {opt}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="标准 MCQ RLCD 训练（CIFAR-10）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", default="output/cifar10_mcq_rlcd")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma-max", type=float, default=0.2)
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--w-sph", type=float, default=0.75)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--pure-ce", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0, help="0=all")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from .dataset import CIFAR10LetterSlot, make_cifar10_scenario
    from .grpo_trainer import GRPOLoss, sigma_schedule
    from .transformers_scorer import find_subsequence

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    scenario = make_cifar10_scenario()
    train_ds = CIFAR10LetterSlot(root=args.data_root, split="train",
                                  augment_order=True, seed=args.seed)
    print(f"train={len(train_ds)}")

    # 模型 + LoRA
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=r".*\.linear$",
        lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    loss_fn = GRPOLoss(G=args.G, sigma=args.sigma_max,
                        w_sph=args.w_sph, lambda_ce=args.lambda_ce)

    anchor_ids = tokenizer.encode("Answer:", add_special_tokens=False)
    total_samples = len(train_ds) * args.epochs
    total_steps = total_samples // args.micro_batch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    log_file = out_dir / "train_log.jsonl"
    step = 0
    t_start = time.time()
    print(f"Total steps: {total_steps}")

    for epoch in range(args.epochs):
        indices = torch.randperm(len(train_ds)).tolist()
        for idx_pos in range(0, len(indices), args.micro_batch):
            if step >= total_steps:
                break

            model.train()
            idx = indices[idx_pos]

            img, label, meta = train_ds[idx]
            gold_idx = meta["gold_idx"]
            options = meta["ordered"][:10]  # 10 classes, no abstain for MCQ

            # 标准 MCQ prompt
            q = scenario.questions[0]
            mcq_text = build_mcq_prompt(q.instructions, options)

            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": mcq_text},
            ]}]
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)

            inputs = processor(text=text, images=[img],
                               return_tensors="pt").to(model.device)

            K = len(options)
            gold = torch.zeros(1, K, device=model.device)
            gold[0, gold_idx] = 1.0

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits

            ids = inputs.input_ids[0].tolist()
            pos = find_subsequence(ids, anchor_ids)
            if pos < 0:
                step += 1
                continue

            mu = logits[0, pos].float().unsqueeze(0)  # [1, vocab]

            # 字母 token ids（上下文解析）
            letter_ids = []
            for j in range(K):
                L = chr(ord("A") + j)
                full = tokenizer.encode("Answer: " + L, add_special_tokens=False)
                suffix = full[len(anchor_ids):]
                letter_ids.append(suffix[0] if suffix else tokenizer.encode(L)[0])
            mu = mu[:, letter_ids]  # [1, K]

            sigma = 0.0
            if args.pure_ce:
                loss = F.cross_entropy(mu, gold)
            else:
                sigma = sigma_schedule(step, total_steps,
                                        args.sigma_max, args.sigma_min)
                loss_fn.sigma = sigma
                loss = loss_fn(mu, gold)
            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                elapsed = time.time() - t_start
                rec = {"step": step, "loss": round(float(loss.item()), 4),
                       "sigma": round(sigma, 4), "epoch": epoch,
                       "elapsed_s": round(elapsed, 1)}
                print(f"step {step}/{total_steps} loss={rec['loss']} "
                      f"σ={rec['sigma']} t={elapsed:.0f}s")
                with open(log_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        if step >= total_steps:
            break

    print("Saving LoRA adapter...")
    model.save_pretrained(str(out_dir / "lora_adapter"))
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
