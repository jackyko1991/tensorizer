"""Checks written first (TDD): what any correct light-cone string code must satisfy.

Run with `pytest problems/cosmology/string_modes/test.py`. Each test is a law of the (bosonic) Nambu–Goto string in
light-cone gauge: the embedding solves the wave equation, the level computed from the field equals the oscillator sum,
level matching is σ-translation invariance, the level degeneracies are the coefficients of ∏(1 − qⁿ)^−24, and the
closed-string spectrum on a circle is invariant under T-duality R ↔ α′/R.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

F64 = torch.float64


def _state(kind, n_a=2, n_b=3):
    alpha, alpha_t = tensors.oscillators(n_a, n_b, kind)
    sigma = tensors.sigma_grid(256, kind, torch.device("cpu"))
    return alpha, alpha_t, sigma


def test_partition_function_gives_the_level_degeneracies():
    d = equation.level_degeneracies(8, colours=24)
    # coefficients of ∏ (1 − qⁿ)^−24: the number of states at level N of the open bosonic string
    known = [1, 24, 324, 3200, 25650, 176256, 1073720, 5930496, 30178575]
    assert d.round().long().tolist() == known
    # one colour: ∏ (1 − qⁿ)^−1 = Σ p(N) qᴺ, the partition numbers
    assert equation.level_degeneracies(10, colours=1).round().long().tolist() == [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]


def test_partition_function_agrees_with_brute_force_count():
    # slow reference: count multisets of (mode n, colour c) with Σ n = N directly with integer arithmetic
    n_max, colours = 6, 24
    ways = [1] + [0] * n_max
    for n in range(1, n_max + 1):  # each of the 24 colours of mode n is an independent oscillator
        for _ in range(colours):
            for level in range(n, n_max + 1):
                ways[level] += ways[level - n]
    assert equation.level_degeneracies(n_max, colours).round().long().tolist() == ways


@pytest.mark.parametrize("kind", ["closed", "open"])
def test_embedding_solves_the_wave_equation(kind):
    alpha, alpha_t, _ = _state(kind)
    h = 1e-3
    tau = torch.tensor([0.3 - h, 0.3, 0.3 + h], dtype=F64)
    sigma = torch.linspace(0.2, 2.5, 40, dtype=F64)
    s3 = torch.stack([sigma - h, sigma, sigma + h], dim=1).flatten()
    X = equation.embedding(tau, s3, alpha, alpha_t, kind).reshape(3, 40, 3, 3)  # (τ, σ, σ±h, dim)
    X_tt = (X[2, :, 1] - 2 * X[1, :, 1] + X[0, :, 1]) / h**2
    X_ss = (X[1, :, 2] - 2 * X[1, :, 1] + X[1, :, 0]) / h**2
    assert torch.allclose(X_tt, X_ss, atol=1e-4)  # light-cone gauge: (∂τ² − ∂σ²) Xⁱ = 0


@pytest.mark.parametrize("kind", ["closed", "open"])
def test_analytic_derivatives_match_finite_differences(kind):
    alpha, alpha_t, sigma = _state(kind)
    h = 1e-5
    tau = torch.tensor([0.7], dtype=F64)
    X_dot, X_prime = equation.derivatives(tau, sigma, alpha, alpha_t, kind)
    num_dot = (equation.embedding(tau + h, sigma, alpha, alpha_t, kind) -
               equation.embedding(tau - h, sigma, alpha, alpha_t, kind)) / (2 * h)  # fmt: skip
    num_prime = (equation.embedding(tau, sigma + h, alpha, alpha_t, kind) -
                 equation.embedding(tau, sigma - h, alpha, alpha_t, kind)) / (2 * h)  # fmt: skip
    assert torch.allclose(X_dot, num_dot, atol=1e-7)
    assert torch.allclose(X_prime, num_prime, atol=1e-7)


@pytest.mark.parametrize("kind", ["closed", "open"])
def test_level_from_field_energy_equals_oscillator_sum(kind):
    alpha, alpha_t, sigma = _state(kind)
    tau = torch.linspace(0, 2 * math.pi, 7, dtype=F64)
    X_dot, X_prime = equation.derivatives(tau, sigma, alpha, alpha_t, kind)
    level = equation.level_from_field(X_dot, X_prime, sigma, kind)  # (T,)
    expected = equation.oscillator_level(alpha) + (equation.oscillator_level(alpha_t) if kind == "closed" else 0)
    # (1/4πα′) ∫ (Ẋ² + X′²) dσ = Σ_n |α_n|² (+ left movers): Parseval, conserved in τ
    assert torch.allclose(level, torch.full_like(level, float(expected)), rtol=1e-10)
    # each quantum in mode n carries n units: N = Σ n k_n, so α′M² rises by n (open) or 2n per side (closed)
    if kind == "open":
        assert expected == pytest.approx(2 + 3)
    else:
        assert equation.oscillator_level(alpha) == pytest.approx(6) == equation.oscillator_level(alpha_t)


def test_level_matching_is_sigma_translation_invariance():
    alpha, alpha_t, sigma = _state("closed")
    tau = torch.linspace(0, 3, 5, dtype=F64)
    X_dot, X_prime = equation.derivatives(tau, sigma, alpha, alpha_t, "closed")
    P_sigma = equation.sigma_momentum(X_dot, X_prime, sigma)
    N, N_t = equation.oscillator_level(alpha), equation.oscillator_level(alpha_t)
    assert N == pytest.approx(N_t)  # the state was built level matched
    assert P_sigma.abs().max() < 1e-10  # ∫ Ẋ·X′ dσ = 2πα′(Ñ − N) = 0
    # an unmatched state has P_σ = 2πα′(Ñ − N) ≠ 0
    X_dot2, X_prime2 = equation.derivatives(tau, sigma, alpha, 0 * alpha_t, "closed")
    P2 = equation.sigma_momentum(X_dot2, X_prime2, sigma)
    assert torch.allclose(P2, torch.full_like(P2, 2 * math.pi * tensors.ALPHA_PRIME * (0 - N)), rtol=1e-10)


def test_closed_string_is_periodic_in_sigma_and_tau():
    alpha, alpha_t, sigma = _state("closed")
    tau = torch.tensor([0.4, 0.4 + 2 * math.pi], dtype=F64)
    X = equation.embedding(tau, torch.tensor([0.1, 0.1 + 2 * math.pi], dtype=F64), alpha, alpha_t, "closed")
    assert torch.allclose(X[0, 0], X[0, 1], atol=1e-12)  # closed loop: X(σ + 2π) = X(σ)
    assert torch.allclose(X[0, 0], X[1, 0], atol=1e-12)  # oscillators only: period 2π in τ (rest frame)


def test_open_string_ends_obey_neumann_condition():
    alpha, alpha_t, _ = _state("open")
    _, X_prime = equation.derivatives(torch.linspace(0, 3, 9, dtype=F64), torch.tensor([0.0, math.pi], dtype=F64),
                                      alpha, alpha_t, "open")  # fmt: skip
    assert X_prime.abs().max() < 1e-12  # free ends: X′ = 0 at σ = 0, π (they move at the speed of light)


def test_mass_formulae_and_intercept():
    N = torch.arange(4, dtype=F64)
    assert equation.open_mass_squared(N).tolist() == [-1.0, 0.0, 1.0, 2.0]  # α′M² = N − 1: tachyon, photon, …
    assert equation.closed_mass_squared(N).tolist() == [-4.0, 0.0, 4.0, 8.0]  # α′M² = 4(N − 1), N = Ñ
    # the 24 states at N = 1 are exactly the D − 2 = 24 transverse polarisations of a massless vector
    assert equation.level_degeneracies(1)[1].item() == pytest.approx(24)


@pytest.mark.parametrize("R", [0.37, 0.8, 1.9, 3.3])
def test_t_duality_of_the_circle_spectrum(R):
    radii = torch.tensor([R, tensors.ALPHA_PRIME / R], dtype=F64)
    M2, weight = solve.circle_spectrum(radii, k_max=6, n_max=3)  # (2, S), (S,)
    # same masses with the same multiplicities: sort each spectrum (states with |n|,|w| ≤ 6 map into each other)
    a, ia = M2[0].sort()
    b, ib = M2[1].sort()
    assert torch.allclose(a, b, atol=1e-10)
    assert torch.equal(weight[ia].sort().values, weight[ib].sort().values)


def test_self_dual_radius_has_extra_massless_states():
    M2, weight = solve.circle_spectrum(torch.tensor([1.0, 1.3], dtype=F64), k_max=3, n_max=2)
    massless = (M2.abs() < 1e-12).to(F64) @ weight
    # at R = √α′ the states n = w = ±1, (N, Ñ) = (1, 0) or (0, 1) become massless: the SU(2)×SU(2) enhancement
    assert massless[0] > massless[1]
