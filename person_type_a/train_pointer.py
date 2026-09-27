"""指针头训练 CLI：交叉注意力式选项打分（绕过字母映射）。

用法（仓库根目录）：
  uv run --with torch --with transformers --with peft --with pillow --with torchvision \
      python -m person_type_a.train_pointer \
      --model /path/to/gemma-4-E4B-it --output-dir output/cifar10_pointer \
      --epochs 1 --micro-batch 1 --grad-accum 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="指针头训练（CIFAR-10）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", default="output/cifar10_pointer")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
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
    from .pointer_head import PointerHead, build_pointer_prompt, find_marker_positions
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

    # 获取 hidden size（从 config 或模型参数推断）
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 2560
    print(f"hidden_size={hidden_size}")

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=r".*\.linear$",
        lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    # 指针头训练不用 gradient checkpointing（与 output_hidden_states 冲突，
    # gemma-4-E4B 足够小，不需要 checkpointing 省 memory）

    # 指针头（独立可训模块，与 LoRA 同设备）
    pointer = PointerHead(hidden_size).to(model.device).to(torch.bfloat16)
    model.print_trainable_parameters()
    print(f"Pointer head params: {sum(p.numel() for p in pointer.parameters())}")

    # 优化器：LoRA + pointer head 分组学习率
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": pointer.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)

    # 标记 token
    opt_marker_ids = tokenizer.encode("</opt>", add_special_tokens=False)
    decide_marker_ids = tokenizer.encode("<decide>", add_special_tokens=False)

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
            pointer.train()
            idx = indices[idx_pos]

            img, label, meta = train_ds[idx]
            gold_idx = meta["gold_idx"]
            options = meta["ordered"]

            # 构造指针头 prompt
            q = scenario.questions[0]
            instructions = q.instructions
            pointer_text = build_pointer_prompt(instructions, options)

            # chat 模板
            from .prompt import build_system_text, chat_messages
            msgs = chat_messages(
                build_system_text(scenario.system, scenario.scene, scenario.evidence),
                pointer_text, img)
            try:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)

            inputs = processor(text=text, images=[img],
                               return_tensors="pt").to(model.device)

            # 前向（取 hidden_states 而非 logits）
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True)
                # 最后一层 hidden states: [1, seq_len, hidden]
                hidden = outputs.hidden_states[-1][0]  # [seq, hidden]

            ids = inputs.input_ids[0].tolist()

            # 找到各选项 </opt> 位置和 <decide> 位置
            opt_positions = find_marker_positions(ids, opt_marker_ids)
            decide_positions = find_marker_positions(ids, decide_marker_ids)

            if len(opt_positions) < len(options) or len(decide_positions) < 1:
                step += 1
                continue

            # 提取选项隐状态 [n_options, hidden]
            opt_hiddens = torch.stack([hidden[pos] for pos in opt_positions])
            # 提取 decide 隐状态 [hidden]
            decide_hidden = hidden[decide_positions[-1]]

            # 指针头打分 → 选项分布
            probs = pointer(
                opt_hiddens.unsqueeze(0).float(),
                decide_hidden.unsqueeze(0).float()
            ).squeeze(0)  # [n_options]

            # CE loss
            gold = torch.tensor(gold_idx, device=probs.device)
            loss = F.cross_entropy(probs.unsqueeze(0), gold.unsqueeze(0))
            loss = loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    lora_params + list(pointer.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                elapsed = time.time() - t_start
                rec = {"step": step, "loss": round(float(loss.item()), 4),
                       "epoch": epoch, "elapsed_s": round(elapsed, 1)}
                print(f"step {step}/{total_steps} loss={rec['loss']} t={elapsed:.0f}s")
                with open(log_file, "a") as f:
                    f.write(json.dumps(rec) + "\n")

        if step >= total_steps:
            break

    # 保存
    print("Saving LoRA + pointer head...")
    model.save_pretrained(str(out_dir / "lora_adapter"))
    torch.save(pointer.state_dict(), str(out_dir / "pointer_head.pt"))
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
