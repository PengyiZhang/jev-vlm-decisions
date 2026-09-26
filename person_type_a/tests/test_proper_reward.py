import pytest

torch = pytest.importorskip("torch")

from person_type_a.proper_reward import proper_reward, spherical_score


def test_perfect_prediction_maximizes_reward():
    """q == y 时 reward 应最大（严格 proper）。"""
    gold = torch.tensor([0.0, 0.0, 1.0])
    r_good = proper_reward(torch.tensor([0.05, 0.05, 0.9]), gold)
    r_bad = proper_reward(torch.tensor([0.9, 0.05, 0.05]), gold)
    assert r_good > r_bad


def test_properness_numerical_grid():
    """数值网格验证：随机扰动 gold 不降低 reward。"""
    torch.manual_seed(42)
    gold = torch.tensor([0.1, 0.3, 0.6])
    r_optimal = proper_reward(gold, gold)
    for _ in range(50):
        noise = torch.randn(3) * 0.1
        perturbed = torch.softmax(torch.log(gold.clamp(min=1e-12)) + noise, dim=-1)
        assert proper_reward(perturbed, gold) <= r_optimal + 1e-4


def test_spherical_scale_invariant():
    gold = torch.tensor([0.0, 1.0])
    aligned = torch.tensor([0.1, 0.9])
    aligned_scaled = aligned * 5
    assert abs(spherical_score(aligned, gold) - spherical_score(aligned_scaled, gold)) < 1e-4


def test_batch_shape():
    gold = torch.eye(3)
    probs = torch.softmax(torch.randn(3, 3), dim=-1)
    r = proper_reward(probs, gold)
    assert r.shape == (3,)
