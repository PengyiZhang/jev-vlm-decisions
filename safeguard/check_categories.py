"""核对数据集类别词表 vs 代码 ALL_CATEGORIES（一次性诊断）。"""
import json
import sys
from collections import Counter

BASE = "/data0/zhangpengyi/Projects/SafeGuardElderlyDomain/data/ElderDomain"
FILES = {
    "train": f"{BASE}/hk_safeguard_train_subset1.jsonl",
    "val": f"{BASE}/hk_safeguard_val_subset1.jsonl",
}

ALL_CATEGORIES = (
    "Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts",
    "PII", "Suicide & Self-Harm", "Unethical Acts",
    "Politically Sensitive Topics", "Copyright Violation", "Jailbreak",
    "HK Welfare & Financial Scam", "RCHE & Caregiver Malpractice",
    "Medication & Health Misguidance", "Hidden Elder Crisis",
)

for name, path in FILES.items():
    label_cats, field_cats = Counter(), Counter()
    n = 0
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            n += 1
            for ln in s["assistant_label"].split("\n"):
                if ln.startswith("Categories:"):
                    body = ln[len("Categories:"):].strip()
                    label_cats.update(c.strip() for c in body.split(",") if c.strip())
            fc = s.get("category", "")
            if fc:
                field_cats[fc] += 1
    print(f"== {name} (n={n}) ==")
    print("assistant_label 类别词表:")
    for c, k in label_cats.most_common():
        mark = "OK " if c in ALL_CATEGORIES else ">>> 不在 ALL_CATEGORIES!"
        print(f"  {mark} {c}: {k}")
    print("category 字段词表:")
    for c, k in field_cats.most_common():
        mark = "OK " if c in ALL_CATEGORIES else ">>> 不在 ALL_CATEGORIES!"
        print(f"  {mark} {c}: {k}")
    missing = [c for c in ALL_CATEGORIES if c not in label_cats]
    print(f"ALL_CATEGORIES 中数据未出现的类别: {missing or '无'}")
    print()
