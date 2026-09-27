"""标准 MCQ 格式零样本评测（支持 --model 切换）。"""
import argparse
import sys
import time

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data-root", default="/tmp/cifar10")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from person_type_a.dataset import CIFAR10LetterSlot, CIFAR10_CLASSES
    from person_type_a.transformers_scorer import find_subsequence

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    ds = CIFAR10LetterSlot(root=args.data_root, split="val",
                           augment_order=False, seed=42)
    anchor_ids = tokenizer.encode("Answer:", add_special_tokens=False)

    correct = 0
    total = 0
    confs, hits = [], []
    t0 = time.perf_counter()

    for i in range(min(args.n, len(ds))):
        img, label, meta = ds[i]
        gold_class = CIFAR10_CLASSES[label]

        options = meta["ordered"][:10]
        letters = [chr(ord("A") + j) for j in range(len(options))]
        if gold_class not in options:
            continue
        gold_idx = options.index(gold_class)

        opt_lines = "\n".join(
            f"{letters[j]}. {options[j]}" for j in range(len(options)))
        question = (f"What object is shown in this image?\n\n"
                    f"{opt_lines}\n\nAnswer:")

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question},
        ]}]
        text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)

        inputs = processor(text=text, images=[img],
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]

        ids = inputs.input_ids[0].tolist()
        pos = find_subsequence(ids, anchor_ids)
        if pos < 0:
            continue

        letter_ids = []
        for L in letters:
            full = tokenizer.encode("Answer: " + L, add_special_tokens=False)
            suffix = full[len(anchor_ids):]
            lid = suffix[0] if suffix else tokenizer.encode(L)[0]
            letter_ids.append(lid)

        scores = logits[pos].float()[letter_ids]
        probs = torch.softmax(scores, dim=-1)
        pred_idx = probs.argmax().item()
        pred_class = options[pred_idx]
        is_correct = pred_class == gold_class
        correct += is_correct
        total += 1
        confs.append(probs.max().item())
        hits.append(1.0 if is_correct else 0.0)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n} acc={correct/total:.3f}")

    dt = time.perf_counter() - t0
    acc = correct / total
    avg_conf = sum(confs) / len(confs)

    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = [j for j, c in enumerate(confs) if lo < c <= hi]
        if mask:
            bin_acc = sum(hits[j] for j in mask) / len(mask)
            bin_conf = sum(confs[j] for j in mask) / len(mask)
            ece += len(mask) / len(confs) * abs(bin_acc - bin_conf)

    print()
    print(f"== Standard MCQ (zero-shot, {args.model.split('/')[-1]}) ==")
    print(f"  samples:  {total}")
    print(f"  accuracy: {acc:.4f}")
    print(f"  avg_conf: {avg_conf:.4f}")
    print(f"  ECE:      {ece:.4f}")
    print(f"  time:     {dt:.0f}s ({dt/total*1000:.0f}ms/sample)")


if __name__ == "__main__":
    main()
