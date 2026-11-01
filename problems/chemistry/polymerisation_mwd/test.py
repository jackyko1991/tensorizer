"""Physics checks, written first (TDD): what any correct polymerisation molecular-weight-distribution code must satisfy.

Run with `pytest problems/chemistry/polymerisation_mwd/test.py`. Each test is a physical law, not an implementation
detail: step growth gives Flory with Carothers' mean, living chains give Poisson, and monomer units are conserved.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.mark.parametrize("p", [0.5, 0.9, 0.98])
def test_flory_distribution_is_normalised_with_carothers_mean_and_dispersity_one_plus_p(p):
    x = tensors.lengths(5000, CPU)
    n, w = equation.flory_number(x, p), equation.flory_weight(x, p)
    assert n.sum().item() == pytest.approx(1.0, abs=1e-9)  # normalised distribution
    assert w.sum().item() == pytest.approx(1.0, abs=1e-9)
    xn = (x * n).sum().item()
    assert xn == pytest.approx(equation.carothers(p), rel=1e-9)  # Carothers: X_n = 1 / (1 − p)
    xn_, xw, disp = equation.averages(n, x)
    assert disp.item() == pytest.approx(1 + p, rel=1e-9)
    assert xw.item() == pytest.approx((1 + p) / (1 - p), rel=1e-9)


def test_population_balance_reproduces_flory_and_poisson():
    L = 400
    idx, mask = tensors.shift_index(L, CPU)
    x = tensors.lengths(L, CPU)
    s, traj = solve.population_balance(tensors.initial_state(L, CPU), idx, mask, 9.0, 0.1)  # X_n = 10, p = 0.9
    P_step, P_living = traj[-1]
    p = s[-1].item() / (1 + s[-1].item())
    assert p == pytest.approx(0.9)
    assert torch.allclose(P_step, (1 - p) ** 2 * p ** (x - 1), atol=1e-6)
    assert torch.allclose(P_living, equation.poisson(x, s[-1]), atol=1e-6)
    m = equation.moments(traj, x)  # (T, 2, 3)
    assert torch.allclose(m[:, 0, 1], torch.ones_like(m[:, 0, 1]), atol=1e-8)  # monomer units conserved
    assert torch.allclose(m[:, 1, 0], torch.ones_like(m[:, 1, 0]), atol=1e-8)  # living chains conserved
    _, _, disp = equation.averages(traj[-1], x)
    assert disp[0].item() == pytest.approx(1 + p, abs=1e-4)


def test_kinetic_monte_carlo_matches_flory():
    n0, p = 50_000, 0.9
    counts = solve.kinetic_monte_carlo(n0, [int(p * n0)], 400, seed=0)[0]  # chains per initial monomer
    x = tensors.lengths(400, CPU)
    xn, _, disp = equation.averages(counts, x)
    assert xn.item() == pytest.approx(1 / (1 - p), rel=1e-9)  # exact: every event removes one chain
    assert disp.item() == pytest.approx(1 + p, abs=0.05)
    cumulative = counts.cumsum(0) / counts.sum()  # fraction of chains no longer than x
    for k in (1, 3, 10, 30):
        assert cumulative[k - 1].item() == pytest.approx(1 - p**k, abs=0.03)  # Flory: Σ_{x≤k} (1-p) p^(x-1)


def test_living_polymerisation_gives_poisson_dispersity():
    L = 300
    idx, mask = tensors.shift_index(L, CPU)
    x = tensors.lengths(L, CPU)
    s, traj = solve.population_balance(tensors.initial_state(L, CPU), idx, mask, 49.0, 0.1)
    xn, _, disp = equation.averages(traj[:, 1], x)  # living row
    nu = s
    assert torch.allclose(xn, 1 + nu, atol=1e-6)
    assert torch.allclose(disp, 1 + nu / (1 + nu) ** 2, atol=1e-6)  # Poisson: Đ = 1 + ν / (1 + ν)²
    assert disp[-1].item() == pytest.approx(1 + 1 / xn[-1].item(), abs=1.01 / xn[-1].item() ** 2)
    assert disp[-1].item() < 1.03
