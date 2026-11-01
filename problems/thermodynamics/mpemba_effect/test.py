"""Physics checks, written first (TDD): what any correct Markovian Mpemba code must satisfy.

Run with `pytest problems/thermodynamics/mpemba_effect/test.py`. The rate matrix conserves probability and obeys
detailed balance, the stationary state is Boltzmann, the spectral solution equals the matrix exponential, and a hot
start whose slow-mode amplitude vanishes overtakes a warm one.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def mpemba():
    return solve.build(barrier=2.0, T_bath=1.0, device=CPU)


def test_rate_matrix_conserves_probability_and_is_metropolis(mpemba):
    W, u = mpemba["W"], mpemba["u"]
    assert W.sum(dim=0).abs().max() < 1e-14  # every column sums to zero: probability is conserved
    off = W - torch.diag(torch.diag(W))
    assert (off >= 0).all() and (torch.diag(W) <= 0).all()
    # detailed balance with the Boltzmann weights: W_ij π_j = W_ji π_i
    pi = mpemba["pi"]
    flux = W * pi[None, :]
    assert torch.allclose(flux, flux.T, atol=1e-15)
    assert torch.allclose(pi, torch.softmax(-u, 0))  # T_b = 1


def test_stationary_state_is_boltzmann(mpemba):
    W, lam, right = mpemba["W"], mpemba["lam"], mpemba["right"]
    assert abs(lam[0]) < 1e-12 and (lam[1:] < 0).all()  # one steady state, every other mode decays
    stationary = right[:, 0] / right[:, 0].sum()
    assert torch.allclose(stationary, mpemba["pi"], atol=1e-12)
    assert (W @ mpemba["pi"]).abs().max() < 1e-15
    # long-time relaxation from any start ends at π
    p0 = tensors.boltzmann(mpemba["u"], torch.tensor([5.0], dtype=torch.float64))
    p_end = torch.linalg.matrix_exp(W * 50 / -lam[1].item()) @ p0[0]
    assert torch.allclose(p_end, mpemba["pi"], atol=1e-10)


def test_spectral_solution_matches_matrix_exponential(mpemba):
    T0 = torch.tensor([1.5, 3.0, 10.0], dtype=torch.float64)
    p0 = tensors.boltzmann(mpemba["u"], T0)
    a = equation.mode_amplitudes(p0, mpemba["left"])
    times = torch.tensor([0.0, 10.0, 300.0], dtype=torch.float64)
    p = equation.evolve(a, mpemba["lam"], mpemba["right"], times)  # (T, B, N)
    for k, t in enumerate(times):
        expected = torch.einsum("ij,bj->bi", torch.linalg.matrix_exp(mpemba["W"] * t), p0)
        assert torch.allclose(p[k], expected, atol=1e-12)  # master equation: p(t) = e^{Wt} p(0)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)))  # normalisation at every time


def test_a2_changes_sign_at_mpemba_point(mpemba):
    u, left = mpemba["u"], mpemba["left"]
    T_star = solve.mpemba_point(u, 1.0, left)
    assert 2.0 < T_star < 4.0
    a2 = solve.a2_curve(u, torch.tensor([T_star / 1.05, T_star, T_star * 1.05], dtype=torch.float64), left)
    assert a2[0] * a2[2] < 0  # opposite signs on either side
    assert abs(a2[1]) < 1e-10
    assert abs(solve.a2_curve(u, torch.tensor([1.0], dtype=torch.float64), left)[0]) < 1e-12  # a_2(T_b) = 0


def test_hot_start_overtakes_warm_start(mpemba):
    u, lam, right, left, pi = (mpemba[k] for k in ["u", "lam", "right", "left", "pi"])
    T_star = solve.mpemba_point(u, 1.0, left)
    T0 = torch.tensor([1.5, T_star, 3.5], dtype=torch.float64)  # warm, strong-Mpemba, hot
    a = equation.mode_amplitudes(tensors.boltzmann(u, T0), left)
    assert abs(a[2, 1]) < abs(a[0, 1])  # the hotter start has the smaller slow-mode amplitude
    times = torch.tensor([0.0, 3000.0], dtype=torch.float64)
    kl = equation.kl_divergence(equation.evolve(a, lam, right, times), pi)  # (T, B)
    assert kl[0, 2] > kl[0, 0]  # starts farther from equilibrium ...
    assert kl[1, 2] < kl[1, 0]  # ... but ends closer
    assert kl[1, 1] < 1e-3 * kl[1, 0]  # at T* the slow mode is absent: exponentially faster
