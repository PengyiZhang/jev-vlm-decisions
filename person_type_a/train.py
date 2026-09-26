"""GRPO 字母槽训练 CLI（CIFAR-10 × LoRA）。

用法（仓库根目录）：
  uv run --with torch --with transformers --with peft --with pillow --with torchvision \
      python -m person_type_a.train \
      --model /path/to/gemma-4-E4B-it --output-dir output/cifar10 \
      --epochs 2 --micro-batch 4 --grad-accum 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO 字母槽训练（CIFAR-10）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", default="output/cifar10")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma-max", type=float, default=0.4)
    ap.add_argument("--sigma-min", type=float, default=0.1)
    ap.add_argument("--w-sph", type=float, default=0.75)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0, help="0=all")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from .dataset import CIFAR10LetterSlot, make_cifar10_scenario
    from .encoding import assign_letters
    from .grpo_trainer import GRPOLoss, sigma_schedule
    from .prompt import build_system_text, build_question_text_single, chat_messages
    from .schema import QuestionSpec
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
    # gemma-4 的投影层被 Gemma4ClippableLinear 包装，PEFT 不识别外层类，
    # 用正则匹配内部 .linear 子层（q/k/v/o_proj + gate/up/down_proj 的内层）
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear$",
        lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    loss_fn = GRPOLoss(G=args.G, sigma=args.sigma_max,
                        w_sph=args.w_sph, lambda_ce=args.lambda_ce)

    anchor_ids = tokenizer.encode("答案：", add_special_tokens=False)
    total_samples = len(train_ds) * args.epochs
    total_steps = total_samples // (args.micro_batch * args.grad_accum)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    log_file = out_dir / "train_log.jsonl"
    step = 0
    t_start = time.time()
    print(f"Total steps: {total_steps}")

    for epoch in range(args.epochs):
        indices = torch.randperm(len(train_ds)).tolist()
        for bs in range(0, len(indices) - args.micro_batch + 1, args.micro_batch):
            if step >= total_steps:
                break

            model.train()
            batch = indices[bs:bs + args.micro_batch]

            prompts, golds, images, metas = [], [], [], []
            for idx in batch:
                img, label, meta = train_ds[idx]
                q = scenario.questions[0]
                ordered_q = QuestionSpec(
                    qid=q.qid, kind=q.kind, instructions=q.instructions,
                    options=tuple(meta["ordered"]),
                    criteria=tuple("" for _ in meta["ordered"]))
                qtext = build_question_text_single(ordered_q)
                msgs = chat_messages(
                    build_system_text(scenario.system, scenario.scene,
                                       scenario.evidence), qtext, img)
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=False,
                        enable_thinking=False)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=False)
                prompts.append(text)
                K = len(meta["ordered"])
                g = torch.zeros(K)
                g[meta["gold_idx"]] = 1.0
                golds.append(g)
                images.append(img)
                metas.append(meta)

            # gemma4 需要逐条处理（processor 输出含 pixel_position_ids 等
            # 额外字段，手动拼接会缺字段），micro_batch 强制为 1
            inputs = processor(text=prompts[0], images=[images[0]],
                               return_tensors="pt").to(model.device)
            gold = torch.stack(golds[:1]).to(model.device)  # [1, K]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits

            mu_batch = []
            ids = inputs.input_ids[0].tolist()
            pos = find_subsequence(ids, anchor_ids)
            if pos < 0:
                pos = len(ids) - 1
            mu = logits[0, pos].float().unsqueeze(0)  # [1, vocab]

            letter_ids = []
            anchor_text = "答案："
            for letter in metas[0]["letters"].values():
                full = tokenizer.encode(anchor_text + letter,
                                          add_special_tokens=False)
                suffix = full[len(anchor_ids):]
                letter_ids.append(suffix[0] if suffix else
                                  tokenizer.encode(letter)[0])
            mu = mu[:, letter_ids]

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
