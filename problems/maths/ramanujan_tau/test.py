"""Mathematical checks, written first (TDD): what any correct code for Ramanujan's τ(n) must satisfy.

Run with `pytest problems/maths/ramanujan_tau/test.py`. Each test is a theorem about τ, not an implementation detail:
the tensor power series agrees with a slow big-integer expansion of q Π (1 − qⁿ)²⁴ and with the classical values,
τ is multiplicative and obeys the Hecke recurrence (Mordell 1917), τ(n) ≡ σ₁₁(n) mod 691 (Ramanujan), the Deligne
bound |τ(p)| ≤ 2 p^{11/2} holds (Deligne 1974), and the normalised values follow the Sato–Tate law (Barnet-Lamb,
Geraghty, Harris and Taylor 2011) closely enough to see it.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _tau_slow(n_max: int) -> list[int]:
    """Reference: expand q Π_{k≥1} (1 − q^k)^24 with Python big ints, one factor (1 − q^k) at a time."""
    c = [1] + [0] * (n_max - 1)  # coefficients of Π (1 − q^k)^24 up to q^{n_max − 1}
    for k in range(1, n_max):
        for _ in range(24):
            for n in range(n_max - 1, k - 1, -1):
                c[n] -= c[n - k]
    return [0] + c  # τ(0) = 0, τ(n) = c[n − 1]


def _primes(n: int) -> list[int]:
    return [p for p in range(2, n + 1) if all(p % d for d in range(2, math.isqrt(p) + 1))]


def test_first_values():
    tau = solve.tau_exact(20, CPU)
    assert tau[1:11].tolist() == [1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920]
    assert tau[11:14].tolist() == [534612, -370944, -577738]


def test_exact_series_matches_slow_expansion():
    assert solve.tau_exact(120, CPU).tolist() == _tau_slow(120)


def test_jacobi_cube_identity():
    """Jacobi: Π (1 − qⁿ)³ = Σ_k (−1)^k (2k + 1) q^{k(k+1)/2}, the sparse seed of the computation."""
    e3 = tensors.eta_cubed(60, CPU, torch.int64)
    slow = [1] + [0] * 59
    for k in range(1, 60):
        for _ in range(3):
            for n in range(59, k - 1, -1):
                slow[n] -= slow[n - k]
    assert e3.tolist() == slow


def test_multiplicative_on_coprime_pairs():
    tau = solve.tau_exact(600, CPU)
    m, n, ok = equation.multiplicativity(tau)
    assert len(m) > 800
    assert ok.all()


def test_hecke_recurrence_on_prime_powers():
    """τ(p^{k+1}) = τ(p) τ(p^k) − p^11 τ(p^{k−1}), checked exactly with Python integers."""
    tau = solve.tau_exact(600, CPU).tolist()
    checked = 0
    for p in _primes(600):
        k = 1
        while p ** (k + 1) <= 600:
            assert tau[p ** (k + 1)] == tau[p] * tau[p**k] - p**11 * tau[p ** (k - 1)]
            k += 1
            checked += 1
    assert checked >= 15


def test_naive_float_fft_loses_the_small_coefficients():
    """Why the residues: a float64 FFT has an absolute error of eps × the largest coefficient (≈ τ(N)), which swamps
    τ(2) = −24 once N is large."""
    e = tensors.eta_cubed(40000, CPU, torch.float64)
    for _ in range(3):
        e = equation.mul_float(e, e)
    assert abs(e[1] + 24) > 1  # τ(2) is the coefficient of q^1 in Π (1 − qⁿ)²⁴


def test_float_series_matches_exact_and_obeys_hecke_far_out():
    exact = solve.tau_exact(600, CPU)
    tau = solve.tau_float(40000, CPU)
    assert torch.allclose(tau[1:601], exact[1:].double(), rtol=1e-12, atol=0)
    p = torch.tensor([q for q in _primes(200) if q * q <= 40000])
    lhs = tau[p * p]
    rhs = tau[p] ** 2 - p.double() ** 11  # τ(p²) = τ(p)² − p^11
    assert ((lhs - rhs).abs() / rhs.abs()).max() < 1e-12


def test_ramanujan_691_congruence():
    """τ(n) ≡ σ₁₁(n) (mod 691), computed exactly on residues."""
    t691 = solve.tau_mod(20000, [691], CPU)[0]
    sigma = equation.sigma_mod(20000, 11, 691, CPU)
    assert torch.equal(t691[1:], sigma[1:])
    exact = solve.tau_exact(600, CPU)
    assert torch.equal(t691[: len(exact)], exact % 691)


def test_deligne_bound_for_all_primes():
    tau = solve.tau_float(100000, CPU)
    primes = tensors.primes_up_to(100000, CPU)
    x = equation.normalise(tau[primes], primes)  # τ(p) / (2 p^{11/2})
    assert len(primes) == 9592
    assert x.abs().max() < 1  # Deligne (1974): |τ(p)| ≤ 2 p^{11/2}


def test_sato_tate_moments():
    """Semicircle moments: E[x²] = 1/4, E[x⁴] = 1/8, odd moments 0 (Catalan numbers C_k / 4^k)."""
    tau = solve.tau_float(100000, CPU)
    primes = tensors.primes_up_to(100000, CPU)
    x = equation.normalise(tau[primes], primes)
    assert abs(x.mean()) < 0.02
    assert abs((x**2).mean() - 0.25) < 0.01
    assert abs((x**4).mean() - 0.125) < 0.01
    exact = equation.semicircle_moments(3)
    assert torch.allclose(exact, torch.tensor([0.25, 0.125, 5 / 64], dtype=torch.float64))
    xs = torch.linspace(-1, 1, 20001, dtype=torch.float64)
    assert abs(torch.trapezoid(equation.semicircle(xs), xs) - 1) < 1e-4
