"""按（问题类型 × 候选数）分桶的温度校准。格式对齐 rlcd-modernbert 的 calibrator 思路。"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .readout import masked_softmax

FORMAT_VERSION = "person-type-calibrator-v1"
GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def bucket_key(kind: str, k: int) -> str:
    return f"{kind}|{k}"


def nll(samples: list[tuple[list[float], int]], temperature: float) -> float:
    """样本为（候选分数, gold 下标）；NLL 越小温度越优。"""
    total = 0.0
    for scores, gold in samples:
        p = masked_softmax(scores, temperature)[gold]
        total -= math.log(max(p, 1e-12))
    return total / len(samples)


def fit_temperature(samples: list[tuple[list[float], int]]) -> float:
    """网格搜索让 NLL 最小的温度（同 jev-starter 门控章节的网格法）。"""
    return min(GRID, key=lambda t: nll(samples, t))


def ece(confs: list[float], hits: list[float], bins: int = 15) -> float:
    """期望校准误差：分桶累计 |实际命中率 - 平均置信度|。"""
    total, out = len(confs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mask = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0)]
        if mask:
            acc = sum(hits[i] for i in mask) / len(mask)
            conf = sum(confs[i] for i in mask) / len(mask)
            out += len(mask) / total * abs(acc - conf)
    return out


def save_calibrator(path: str | Path, buckets: dict[str, float]) -> None:
    Path(path).write_text(
        json.dumps({"format_version": FORMAT_VERSION, "buckets": buckets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibrator(path: str | Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"未知校准器格式: {data.get('format_version')}")
    return data["buckets"]
