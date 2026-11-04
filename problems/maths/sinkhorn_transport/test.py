"""Mathematical checks, written first (TDD): what any correct optimal transport code must satisfy.

Run with `pytest problems/maths/sinkhorn_transport/test.py`. Each test is a theorem or an exact property: the
Sinkhorn plan is non-negative with the prescribed marginals; as ε → 0 its cost decreases to the exact assignment
cost (checked against a brute-force search over all permutations and the Hungarian algorithm); in 1D the optimum
pairs sorted points (the quantile formula); between Gaussians W2 has the closed Bures form; swapping the two
measures transposes the plan; and the Sinkhorn divergence vanishes on equal measures and is positive otherwise.
"""

import itertools

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def uniform(n: int) -> torch.Tensor:
    return torch.full((n,), 1.0 / n, dtype=torch.float64)


def brute_force_assignment(c: torch.Tensor) -> float:
    """min over all n! permutations of (1/n) Σ_i C_{i σ(i)}, as one tensor of permutations."""
    n = c.shape[0]
    perms = torch.tensor(list(itertools.permutations(range(n))))  # (n!, n)
    return (c[torch.arange(n), perms].sum(1).min() / n).item()


def random_clouds(n: int, m: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=gen, dtype=torch.float64), torch.rand(m, 2, generator=gen, dtype=torch.float64)


def test_plan_is_nonnegative_with_the_right_marginals():
    x, y = random_clouds(40, 30, 0)
    gen = torch.Generator().manual_seed(1)
    a = torch.rand(40, generator=gen, dtype=torch.float64) + 0.1
    b = torch.rand(30, generator=gen, dtype=torch.float64) + 0.1
    a, b = a / a.sum(), b / b.sum()
    eps = torch.tensor([0.5, 0.1, 0.02], dtype=torch.float64)  # a batch of ε at once
    out = solve.sinkhorn(a, b, equation.cost_matrix(x, y), eps, n_iter=1000)
    p = out["plan"]  # (E, N, M)
    assert (p >= 0).all()
    assert torch.allclose(p.sum(-1), a.expand(3, -1), atol=1e-9)  # Σ_j P_ij = a_i
    assert torch.allclose(p.sum(-2), b.expand(3, -1), atol=1e-12)  # Σ_i P_ij = b_j


def test_hungarian_matches_brute_force():
    for seed in range(4):
        x, y = random_clouds(7, 7, 10 + seed)
        c = equation.cost_matrix(x, y)
        perm = solve.exact_assignment(c)
        assert sorted(perm.tolist()) == list(range(7))  # a permutation
        assert c[torch.arange(7), perm].mean().item() == pytest.approx(brute_force_assignment(c), rel=1e-12)


def test_cost_decreases_to_the_exact_assignment_as_eps_goes_to_zero():
    x, y = random_clouds(7, 7, 2)
    c = equation.cost_matrix(x, y)
    exact = brute_force_assignment(c)
    eps = torch.tensor([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001], dtype=torch.float64)
    cost = solve.sinkhorn(uniform(7), uniform(7), c, eps, n_iter=3000, eps_start=1.0)["cost"]  # ⟨P_ε, C⟩ (E,)
    assert (cost >= exact - 1e-9).all()  # no coupling beats the optimum
    assert (cost.diff() <= 1e-7).all()  # the cost of the entropic plan decreases with ε
    assert cost[-1].item() == pytest.approx(exact, rel=1e-3)


def test_one_dimensional_w2_is_the_quantile_formula():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(60, 1, generator=gen, dtype=torch.float64)
    y = 0.5 * torch.rand(60, 1, generator=gen, dtype=torch.float64) ** 2 + 1
    quantile = equation.w2_squared_1d(x[:, 0], y[:, 0])  # mean (x_(i) − y_(i))² of the sorted samples
    c = equation.cost_matrix(x, y)
    assert c[torch.arange(60), solve.exact_assignment(c)].mean().item() == pytest.approx(quantile.item(), rel=1e-12)
    eps = torch.tensor([1e-3], dtype=torch.float64)
    cost = solve.sinkhorn(uniform(60), uniform(60), c, eps, n_iter=3000, eps_start=1.0)["cost"]
    assert cost[0].item() == pytest.approx(quantile.item(), rel=5e-3)


def test_gaussian_w2_has_the_bures_closed_form():
    m1, m2 = torch.tensor([0.0, 0.0], dtype=torch.float64), torch.tensor([1.0, -0.5], dtype=torch.float64)
    s1 = torch.tensor([[1.0, 0.3], [0.3, 0.5]], dtype=torch.float64)
    s2 = torch.tensor([[0.4, -0.2], [-0.2, 0.9]], dtype=torch.float64)
    bures = equation.gaussian_w2_squared(m1, s1, m2, s2)
    assert equation.gaussian_w2_squared(m1, s1, m1, s1).abs() < 1e-12  # W2(μ, μ) = 0
    assert bures.item() == pytest.approx(equation.gaussian_w2_squared(m2, s2, m1, s1).item(), rel=1e-10)  # symmetric
    # The linear Monge map T(x) = m2 + A(x − m1) pushes N(m1, S1) to N(m2, S2) and is the gradient of a convex
    # function, so it is optimal (Brenier): E|x − T(x)|² must equal the closed form.
    a = equation.gaussian_monge_matrix(s1, s2)
    assert torch.allclose(a @ s1 @ a.T, s2, atol=1e-12)  # T pushes S1 forward to S2
    x = tensors.gaussian_samples(m1, s1, 20000, CPU, seed=4)
    tx = m2 + (x - m1) @ a.T
    assert (x - tx).pow(2).sum(-1).mean().item() == pytest.approx(bures.item(), rel=0.03)
    # On a small sample paired by T, Sinkhorn with small ε recovers the same (identity) matching.
    xs, txs = x[:120], tx[:120]
    c = equation.cost_matrix(xs, txs)
    out = solve.sinkhorn(uniform(120), uniform(120), c, torch.tensor([2e-3], dtype=torch.float64), 3000, 2.0)
    assert out["cost"][0].item() == pytest.approx((xs - txs).pow(2).sum(-1).mean().item(), rel=0.01)


def test_swapping_the_measures_transposes_the_plan():
    x, y = random_clouds(25, 35, 5)
    a, b = uniform(25), uniform(35)
    c = equation.cost_matrix(x, y)
    eps = torch.tensor([0.2, 0.05], dtype=torch.float64)
    one = solve.sinkhorn(a, b, c, eps, n_iter=2000)
    two = solve.sinkhorn(b, a, c.T, eps, n_iter=2000)
    assert torch.allclose(one["plan"], two["plan"].transpose(1, 2), atol=1e-9)
    assert torch.allclose(one["cost"], two["cost"], atol=1e-10)
    assert torch.allclose(one["dual"], two["dual"], atol=1e-8)  # OT_ε(a, b; C) = OT_ε(b, a; Cᵀ)


def test_sinkhorn_divergence_is_zero_on_equal_measures_and_positive_otherwise():
    x, y = random_clouds(30, 30, 6)
    a = uniform(30)
    eps = torch.tensor([0.5, 0.05], dtype=torch.float64)
    shuffled = x[torch.randperm(30, generator=torch.Generator().manual_seed(7))]  # same measure, other order
    same = solve.sinkhorn_divergence(x, a, shuffled, a, eps, n_iter=1000)
    diff = solve.sinkhorn_divergence(x, a, y + 0.3, a, eps, n_iter=1000)
    assert same.abs().max() < 1e-9
    assert (diff > 0).all()
