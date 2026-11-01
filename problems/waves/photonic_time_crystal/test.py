"""Physics checks, written first (TDD): what any correct photonic-time-crystal code must satisfy.

Run with `pytest problems/waves/photonic_time_crystal/test.py`. A time interface keeps D and B continuous (temporal
Fresnel), the monodromy has determinant 1, and modes grow only inside the momentum gap at the Floquet rate.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_temporal_fresnel_coefficients_keep_D_and_B_continuous():
    for n1, n2 in [(1.0, 2.0), (1.5, 1.0), (1.0, 1.0), (2.0, 3.0)]:
        Df, Db = equation.temporal_fresnel(n1, n2)
        assert math.isclose(Df + Db, 1.0, abs_tol=1e-12)  # D before = D after
        assert math.isclose(Df / n2 - Db / n2, 1.0 / n1, abs_tol=1e-12)  # B before = B after (B = ±D/n)


def test_simulated_time_reflection_matches_fresnel():
    x, k = tensors.grid(CPU)
    D0, B0 = tensors.forward_pulse(x, 1.0, 1.0)
    _, u = solve.interface(D0, B0, k, 1.0, 4.0, 10.0, 40.0, 0.05, 100)
    D_fwd, D_bwd = equation.split_directions(solve.to_field(u[-1]), solve.to_field(u[-1], 1), 2.0)
    Df, Db = equation.temporal_fresnel(1.0, 2.0)
    assert abs(solve.envelope(D_fwd, k).max().item() - abs(Df)) < 0.01
    assert abs(solve.envelope(D_bwd, k).max().item() - abs(Db)) < 0.01


def _floquet(alpha: float, omega: float = 2.0, n: int = 600):
    period = 2 * math.pi / omega
    k = tensors.band_wavenumbers(1.5 * omega / 2, n, CPU)  # first gap only (the second sits near k = Ω)

    def eps(t: float) -> float:
        return equation.periodic_eps(t, 1.0, alpha, omega)

    M = solve.monodromy(k, eps, period)
    w = equation.floquet_frequencies(torch.linalg.eigvals(M), period)
    return k, M, w.imag.max(dim=-1).values


def test_floquet_growth_only_inside_the_gap():
    alpha = 0.1
    k, M, growth = _floquet(alpha)
    assert torch.allclose(torch.linalg.det(M), torch.ones(len(k), dtype=M.dtype), atol=1e-8)  # λ+ λ- = 1
    lo, hi = equation.perturbative_gap(1.0, alpha, 2.0)
    half = (hi - lo) / 2
    inside = (k > lo + 0.3 * half) & (k < hi - 0.3 * half)
    outside = (k < lo - 0.3 * half) | (k > hi + 0.3 * half)
    assert (growth[inside] > 1e-3).all()
    assert (growth[outside] < 1e-6).all()
    assert math.isclose(growth.max().item(), alpha * 2.0 / 8, rel_tol=0.05)  # peak growth ≈ αΩ/8


def test_gap_edges_match_perturbation_theory():
    for alpha in (0.04, 0.08):
        k, _, growth = _floquet(alpha, n=3000)
        in_gap = k[growth > 1e-6]
        lo, hi = equation.perturbative_gap(1.0, alpha, 2.0)
        tol = alpha**2 + 2 * (k[1] - k[0]).item()  # first-order theory: error O(α²), plus the k spacing
        assert abs(in_gap.min().item() - lo) < tol
        assert abs(in_gap.max().item() - hi) < tol


def test_simulated_mode_grows_at_floquet_rate():
    alpha, omega = 0.2, 2.0
    x, k = tensors.grid(CPU)

    def eps(t: float) -> float:
        return equation.periodic_eps(t, 1.0, alpha, omega)

    D0, B0 = tensors.forward_pulse(x, 1.0, math.sqrt(eps(0.0)))
    times, u = solve.evolve(solve.to_modes(D0, B0), k, eps, 0.0, 0.05, 1600, 10)
    i = torch.argmin((k - 1.0).abs()).item()
    rate = solve.growth_rate_measured(u[..., 0].abs(), torch.tensor(times), i, 40.0)
    M = solve.monodromy(k[i : i + 1], eps, 2 * math.pi / omega)
    expected = equation.floquet_frequencies(torch.linalg.eigvals(M), 2 * math.pi / omega).imag.max().item()
    assert math.isclose(rate, expected, rel_tol=0.05)
