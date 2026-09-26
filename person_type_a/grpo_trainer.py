"""GRPO（Group Relative Policy Optimization）训练循环。

适配字母槽：噪声注入在锚位字母子集 logits 上，
策略梯度经 μ_θ 回传到 LoRA 参数。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .proper_reward import proper_reward


def sigma_schedule(step: int, total_steps: int,
                   sigma_max: float = 0.4, sigma_min: float = 0.1) -> float:
    """余弦退火：从 sigma_max 降到 sigma_min。"""
    t = min(step / max(total_steps, 1), 1.0)
    return sigma_min + 0.5 * (sigma_max - sigma_min) * (1 + math.cos(math.pi * t))


def sample_zero_mean_noise(mu: torch.Tensor, sigma: float) -> torch.Tensor:
    """采样零均值高斯噪声（投影到零和，保证选项对称）。"""
    eps = torch.randn_like(mu) * sigma
    return eps - eps.mean(dim=-1, keepdim=True)


class GRPOLoss:
    """GRPO 损失：策略梯度 + 软 CE 引导。"""

    def __init__(self, G: int = 4, sigma: float = 0.4,
                 w_sph: float = 0.75, lambda_ce: float = 1.0):
        self.G = G
        self.sigma = sigma
        self.w_sph = w_sph
        self.lambda_ce = lambda_ce

    def __call__(self, mu: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        """mu: [batch, K] 字母子集 logits；gold: [batch, K] 软目标。"""
        all_rewards, all_log_pis = [], []
        for _ in range(self.G):
            with torch.no_grad():
                eps = sample_zero_mean_noise(mu, self.sigma)
            z = mu + eps
            probs = F.softmax(z, dim=-1)
            reward = proper_reward(probs, gold, self.w_sph)
            log_pi = -((z - mu) ** 2).sum(-1) / (2 * self.sigma ** 2)
            all_rewards.append(reward)
            all_log_pis.append(log_pi)

        rewards = torch.stack(all_rewards)
        log_pis = torch.stack(all_log_pis)
        advantages = (rewards - rewards.mean(0, keepdim=True)) / \
                     (rewards.std(0, keepdim=True) + 1e-8)

        loss_pg = -(advantages.detach() * log_pis).mean()
        loss_ce = F.cross_entropy(mu, gold)
        return loss_pg + self.lambda_ce * loss_ce
