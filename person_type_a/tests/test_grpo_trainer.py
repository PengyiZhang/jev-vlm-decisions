import pytest

torch = pytest.importorskip("torch")

from demos.person_type_a.grpo_trainer import GRPOLoss, sample_zero_mean_noise, sigma_schedule


def test_sigma_schedule_cosine_annealing():
    assert sigma_schedule(0, 100, 0.4, 0.1) == pytest.approx(0.4)
    assert sigma_schedule(50, 100, 0.4, 0.1) == pytest.approx(0.25, abs=0.01)
    assert sigma_schedule(100, 100, 0.4, 0.1) == pytest.approx(0.1, abs=0.01)


def test_noise_is_zero_mean():
    torch.manual_seed(42)
    mu = torch.zeros(100, 11)
    eps = sample_zero_mean_noise(mu, sigma=0.3)
    assert eps.shape == mu.shape
    assert abs(eps.mean(-1).max().item()) < 1e-5


def test_grpo_loss_positive_and_finite():
    torch.manual_seed(0)
    mu = torch.randn(2, 4)
    gold = torch.eye(4)[:2]
    loss_fn = GRPOLoss(G=4, sigma=0.2)
    loss = loss_fn(mu, gold)
    assert loss > 0
    assert torch.isfinite(loss)


def test_policy_gradient_flows_to_mu():
    torch.manual_seed(0)
    mu = torch.randn(2, 4, requires_grad=True)
    gold = torch.eye(4)[:2]
    loss_fn = GRPOLoss(G=4, sigma=0.2)
    loss = loss_fn(mu, gold)
    loss.backward()
    assert mu.grad is not None
    assert mu.grad.abs().max() > 0


def test_policy_gradient_survives_beyond_ce():
    """回归：PG 项自身必须贡献非零梯度（λ_ce=0 时梯度仍非零）。

    历史缺陷：z = mu + eps 使 log_pi 里的 z-mu 全导数为零，
    GRPOLoss 梯度与纯 CE 逐位相同（策略梯度从未生效）。
    """
    torch.manual_seed(0)
    mu = torch.randn(2, 4, requires_grad=True)
    gold = torch.eye(4)[:2]
    loss_fn = GRPOLoss(G=8, sigma=0.2, lambda_ce=0.0)
    loss_fn(mu, gold).backward()
    assert mu.grad is not None
    assert mu.grad.abs().max() > 1e-6


def test_full_loss_gradient_differs_from_pure_ce():
    """回归：总损失梯度 ≠ 纯 CE 梯度（同一 mu、同一 gold 下）。"""
    torch.manual_seed(0)
    mu0 = torch.randn(2, 4)
    gold = torch.eye(4)[:2]

    mu1 = mu0.clone().requires_grad_(True)
    GRPOLoss(G=8, sigma=0.2, lambda_ce=1.0)(mu1, gold).backward()

    mu2 = mu0.clone().requires_grad_(True)
    torch.nn.functional.cross_entropy(mu2, gold).backward()

    assert (mu1.grad - mu2.grad).abs().max() > 1e-6
