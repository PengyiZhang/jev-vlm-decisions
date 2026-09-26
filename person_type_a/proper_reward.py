"""严格 proper scoring rule 组合：log score + spherical score。

移植自 laya（repos/laya/laya/common.py:278-304），适配字母槽 logits。
用于 GRPO 训练的 reward 计算。
"""
from __future__ import annotations

import torch


def log_score(probs: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """对数似然：Σᵢ yᵢ·log(qᵢ)，clamp 防 −inf。"""
    return (gold * torch.log(probs.clamp(min=1e-12))).sum(-1)


def spherical_score(probs: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """球面分数：(Σᵢ yᵢqᵢ) / ‖q‖₂，尺度不变，抗锐化。"""
    return (gold * probs).sum(-1) / probs.norm(dim=-1).clamp(min=1e-12)


def proper_reward(probs: torch.Tensor, gold: torch.Tensor,
                  w_sph: float = 0.75) -> torch.Tensor:
    """choice 类型的严格 proper reward。

    log 与 spherical 均为严格 proper（E_y[r] 在 q=y 处唯一最大），
    非负权重和保持严格 proper——最优策略 = 输出真实概率分布。
    """
    return log_score(probs, gold) + w_sph * spherical_score(probs, gold)
