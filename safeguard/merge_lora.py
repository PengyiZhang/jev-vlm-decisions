"""合并 LoRA adapter 到基座，保存为独立模型目录（消除推理期 LoRA 开销）。"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="合并答案卡 LoRA")
    ap.add_argument("--model", default="/data0/zhangpengyi/LLMs/Qwen3Guard-Gen-0.6B")
    ap.add_argument("--lora", default="safeguard/output/card_rlcd/lora_adapter")
    ap.add_argument("--out", default="safeguard/output/card_rlcd/merged")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.lora)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"merged → {args.out}")


if __name__ == "__main__":
    main()
