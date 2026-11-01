"""Physics checks, written first (TDD): what any correct ζ(s) code must satisfy.

Run with `pytest problems/maths/zeta_primes/test.py`. Each test is a mathematical invariant, not an implementation
detail: ζ takes its closed-form values (π²/6, π⁴/90, −1/2, −1/12), the first zeros lie at the known heights on the
critical line, and the explicit formula rebuilds π(x) from them.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_zeta_at_known_values():
    s = torch.tensor([2.0, 4.0, 0.0, -1.0], dtype=torch.complex128)
    z = equation.zeta(s, 30).real
    expected = torch.tensor([math.pi**2 / 6, math.pi**4 / 90, -0.5, -1 / 12], dtype=torch.float64)
    assert torch.allclose(z, expected, atol=1e-12)  # Euler: ζ(2) = π²/6, ζ(4) = π⁴/90; ζ(0) = −1/2, ζ(−1) = −1/12


def test_first_three_zeta_zeros():
    t = tensors.t_grid(30.0, 0.01, CPU)
    n_terms = solve.n_terms_for(30.0)
    gammas = solve.find_zeros(t, solve.z_on_grid(t, n_terms), n_terms)
    assert len(gammas) == 3
    for g, known in zip(gammas.tolist(), [14.134725142, 21.022039639, 25.010857580]):
        assert abs(g - known) < 1e-4
    # ζ really vanishes there, and Z is real (imaginary part of e^{iθ}ζ ≈ 0) on the critical line
    assert equation.zeta(0.5 + 1j * gammas, n_terms).abs().max() < 1e-8
    t_test = torch.linspace(10, 60, 50, dtype=torch.float64)
    rotated = torch.exp(1j * equation.theta(t_test)) * equation.zeta(0.5 + 1j * t_test, 50)
    assert rotated.imag.abs().max() < 1e-7


def test_riemann_siegel_agrees_with_euler_maclaurin():
    """With only the first correction term the error is O(t^{-3/4}): small, and shrinking with t."""
    low = torch.linspace(50, 100, 200, dtype=torch.float64)
    high = torch.linspace(400, 450, 200, dtype=torch.float64)
    err_low = (equation.riemann_siegel_z(low) - equation.hardy_z(low, 80)).abs().max()
    err_high = (equation.riemann_siegel_z(high) - equation.hardy_z(high, 250)).abs().max()
    assert err_low < 1e-2 and err_high < err_low / 2


def test_sieve_matches_known_prime_counts():
    pi = tensors.prime_pi(tensors.sieve(10**6, CPU))
    assert [int(pi[10**n]) for n in range(1, 7)] == [4, 25, 168, 1229, 9592, 78498]


def test_li_beats_x_over_log_x_at_one_million():
    x = torch.tensor([1e6], dtype=torch.float64)
    pi = 78498
    li = equation.li(x).item()
    assert abs(li - 78627.549159) < 1e-3
    assert abs(li - pi) < abs((x / torch.log(x)).item() - pi)
    r = equation.gram_r(x, tensors.zeta_at_integers(100, CPU)).item()
    assert abs(r - pi) < abs(li - pi)


def test_explicit_formula_improves_with_more_zeros():
    t = tensors.t_grid(tensors.height_for(60), 0.01, CPU)
    n_terms = solve.n_terms_for(t[-1].item())
    gammas = solve.find_zeros(t, solve.z_on_grid(t, n_terms), n_terms)[:60]
    x = torch.arange(2.5, 60, 1.0, dtype=torch.float64)  # half-integers, away from the jumps
    exact = tensors.prime_pi(tensors.sieve(60, CPU))[x.floor().long()].double()
    pi_k = solve.prime_count_from_zeros(x, gammas, tensors.mobius(64))
    err = (pi_k - exact).abs().mean(dim=1)
    assert err[60] < err[10] < err[0]
    assert (pi_k[60] - exact).abs().max() < 0.5  # rounding recovers π(x) exactly
