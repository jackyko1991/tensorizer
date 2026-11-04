"""Checks written first (TDD): what any correct epidemic-on-a-network code must satisfy.

Run with `pytest problems/others/network_sir/test.py`. Each test is a law of the model: people are only moved between
S, I and R, an outbreak below the threshold fizzles, the well-mixed limit obeys the Kermack–McKendrick final-size
equation, the largest eigenvalue of a k-regular graph is k, and vaccinating hubs beats vaccinating at random.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _seed(b, n, n0, gen):
    """n0 random initial infected per realisation, (B, N) bool."""
    order = torch.argsort(torch.rand(b, n, generator=gen), dim=1)
    return order < n0


def test_people_are_conserved():
    gen = torch.Generator().manual_seed(0)
    A = tensors.barabasi_albert(300, 3, gen, CPU)
    beta = torch.full((16,), 0.3)
    for sis in (False, True):
        out = solve.simulate(A, beta, 1.0, 0.1, 200, _seed(16, 300, 5, gen), None, gen, sis=sis)
        total = out["counts"].sum(-1)  # (T, B) S + I + R
        assert torch.all(total == 300)
        assert torch.all(out["counts"] >= 0)
    traj = solve.mean_field(A, beta[:2], 1.0, torch.full((2, 300), 0.01), 0.05, 100)  # (T, 3, B, N)
    assert torch.allclose(traj.sum(1), torch.ones(traj.shape[0], 2, 300, dtype=traj.dtype), atol=1e-9)


def test_regular_graph_eigenvalue_is_its_degree():
    for k in (2, 4, 6):
        A = tensors.ring_lattice(40, k, CPU)
        assert torch.all(A.sum(1) == k)
        assert equation.lambda_max(A).item() == pytest.approx(k, rel=1e-9)
    assert equation.lambda_max(tensors.complete_graph(25, CPU)).item() == pytest.approx(24, rel=1e-9)


def test_below_threshold_the_outbreak_dies_out():
    gen = torch.Generator().manual_seed(1)
    A = tensors.load_workplace(CPU)
    lam = equation.lambda_max(A).item()
    gamma = 1.0
    beta = torch.full((64,), 0.5 * gamma / lam)  # β/γ at half the mean-field threshold 1/λ_max
    out = solve.simulate(A, beta, gamma, 0.1, 600, _seed(64, A.shape[0], 1, gen), None, gen)
    final = out["counts"][-1, :, 2].float() / A.shape[0]  # (B,) fraction ever infected
    assert final.mean() < 0.05 and final.max() < 0.25


def test_well_mixed_limit_matches_final_size_equation():
    gen = torch.Generator().manual_seed(2)
    n, r0, gamma = 400, 2.0, 1.0
    z = equation.final_size(torch.tensor([r0], dtype=torch.float64)).item()
    assert z == pytest.approx(1 - math.exp(-r0 * z), abs=1e-12)  # the root itself
    A = tensors.complete_graph(n, CPU)
    beta = torch.full((96,), r0 * gamma / (n - 1))  # every pair in contact: R0 = β (N − 1) / γ
    out = solve.simulate(A, beta, gamma, 0.05, 1200, _seed(96, n, 4, gen), None, gen)
    final = out["counts"][-1, :, 2].float() / n
    major = final > 0.3  # condition on a major outbreak (a few die out by chance at the start)
    assert major.sum() > 60
    assert final[major].mean().item() == pytest.approx(z, abs=0.03)  # z = 1 − exp(−R0 z) ≈ 0.797
    # the mean-field ODE on the complete graph is the classic well-mixed SIR: same final size
    traj = solve.mean_field(A, beta[:1].double(), gamma, torch.full((1, n), 1e-4, dtype=torch.float64), 0.05, 800)
    assert traj[-1, 2].mean().item() == pytest.approx(z, abs=0.01)


def test_targeted_vaccination_beats_random_on_barabasi_albert():
    gen = torch.Generator().manual_seed(3)
    n = 500
    A = tensors.barabasi_albert(n, 3, gen, CPU)
    lam = equation.lambda_max(A).item()
    beta = torch.full((48,), 4.0 / lam)  # four times the mean-field threshold
    hubs = solve.vaccinate(A, 0.15, "targeted", 48, gen)
    rand = solve.vaccinate(A, 0.15, "random", 48, gen)
    assert hubs.sum(1).eq(75).all() and rand.sum(1).eq(75).all()
    seeds = _seed(48, n, 5, gen)
    f_hubs = solve.simulate(A, beta, 1.0, 0.1, 400, seeds & ~hubs, hubs, gen)["ever"][-1].float().mean()
    f_rand = solve.simulate(A, beta, 1.0, 0.1, 400, seeds & ~rand, rand, gen)["ever"][-1].float().mean()
    assert f_hubs < 0.5 * f_rand  # removing the hubs cuts the paths the epidemic needs
    # and the mean-field threshold moves the same way: λ_max drops much more when the hubs go
    assert equation.lambda_max(solve.remove(A, hubs[0])) < equation.lambda_max(solve.remove(A, rand[0]))


def test_nonbacktracking_radius_of_regular_graph():
    A = tensors.ring_lattice(30, 4, CPU)
    assert equation.lambda_nonbacktracking(A).item() == pytest.approx(3.0, rel=1e-6)  # k − 1 for a k-regular graph
