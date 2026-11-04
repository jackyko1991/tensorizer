"""Mathematical checks, written first (TDD): what any correct Kuramoto code must satisfy.

Run with `pytest problems/others/kuramoto_sync/test.py`. Each test is a known result, not an implementation detail:
below K_c the order parameter is only finite-size noise O(1/√N); above it, for Lorentzian frequencies, r follows the
exact Ott–Antonsen curve √(1 − K_c/K); identical oscillators lock; the mean phase drifts at the mean frequency
(the coupling is antisymmetric); the mean-field form equals the O(N²) pairwise sum; and the ring with phase lag shows
a chimera (a locked and a drifting part side by side).
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_mean_field_equals_pairwise_sum():
    gen = torch.Generator().manual_seed(0)
    theta = 2 * math.pi * torch.rand(3, 50, generator=gen, dtype=torch.float64)
    omega = torch.randn(3, 50, generator=gen, dtype=torch.float64)
    k = torch.tensor([[0.3], [1.0], [2.5]], dtype=torch.float64)
    fast = equation.velocity(theta, omega, k)  # O(N): ω + K r sin(ψ − θ)
    slow = equation.pairwise_velocity(theta, omega, k)  # O(N²): ω + (K/N) Σ_j sin(θ_j − θ_i)
    assert torch.allclose(fast, slow, atol=1e-12)


def test_critical_couplings():
    assert equation.critical_coupling("lorentzian", 0.5) == pytest.approx(1.0)  # K_c = 2γ
    g0 = 1 / (0.7 * math.sqrt(2 * math.pi))  # Gaussian density at 0
    assert equation.critical_coupling("gaussian", 0.7) == pytest.approx(2 / (math.pi * g0))


def test_incoherent_below_critical_coupling():
    n = 1000
    omega = tensors.frequencies("lorentzian", n, 0.5, CPU, seed=1)
    k = torch.tensor([0.0, 0.4, 0.7])[:, None]  # all below K_c = 1
    _, r = solve.sweep(omega, k, t_max=40.0, dt=0.05, seed=2)
    r_late = r[:, r.shape[1] // 2 :].mean(1)  # time average after the transient
    assert (r_late < 4 / math.sqrt(n)).all()  # only finite-size fluctuations O(1/√N)


@pytest.mark.slow
def test_lorentzian_order_parameter_follows_ott_antonsen():
    n = 4000
    gamma = 0.5
    omega = tensors.frequencies("lorentzian", n, gamma, CPU, seed=3)
    k = torch.tensor([1.5, 2.0, 3.0])[:, None]  # K / K_c = 1.5, 2, 3
    _, r = solve.sweep(omega, k, t_max=120.0, dt=0.05, seed=4)
    r_late = r[:, r.shape[1] // 2 :].mean(1)
    exact = equation.ott_antonsen_r(k[:, 0], gamma)  # √(1 − 2γ/K): 0.577, 0.707, 0.816
    assert torch.allclose(r_late, exact.to(r_late), atol=0.03), (r_late, exact)


def test_identical_oscillators_lock():
    omega = torch.full((500,), 0.8)  # everybody has the same frequency
    _, r = solve.sweep(omega, torch.tensor([[0.5]]), t_max=40.0, dt=0.05, seed=5)
    assert r[0, -1] > 0.999  # any K > 0 synchronises them completely


def test_mean_phase_drifts_at_mean_frequency():
    omega = tensors.frequencies("gaussian", 300, 1.0, CPU, seed=6) + 0.3  # nonzero mean frequency
    theta0 = tensors.initial_phases(1, 300, CPU, seed=7)
    theta = solve.integrate(theta0, omega, torch.tensor([[2.0]]), t_max=10.0, dt=0.01)[0]  # (T, B, N)
    drift = theta[-1].mean() - theta[0].mean()  # phases are not wrapped
    assert drift.item() == pytest.approx(omega.mean().item() * 10.0, rel=1e-5)  # Σ sin(θ_j − θ_i) = 0 pairwise


def test_order_parameter_bounds_and_rotation_invariance():
    theta = tensors.initial_phases(4, 100, CPU, seed=8)
    r, psi = equation.order_parameter(theta)
    r2, psi2 = equation.order_parameter(theta + 1.234)  # rotate every phase
    assert ((r >= 0) & (r <= 1)).all()
    assert torch.allclose(r, r2, atol=1e-6)
    assert torch.allclose(torch.remainder(psi2 - psi, 2 * math.pi), torch.full_like(psi, 1.234), atol=1e-5)


@pytest.mark.slow
def test_ring_with_phase_lag_shows_a_chimera():
    _, theta = solve.chimera(n=256, t_max=300.0, dt=0.05, seed=9)
    local = equation.local_order(theta, 5)  # (N,) coherence of each oscillator's neighbourhood
    assert local.max() > 0.95  # a synchronised domain …
    assert local.min() < 0.8  # … next to an incoherent one, although every oscillator is identical
