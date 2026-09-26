import pytest

torch = pytest.importorskip("torch")

from person_type_a.grpo_trainer import GRPOLoss, sample_zero_mean_noise, sigma_schedule


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
