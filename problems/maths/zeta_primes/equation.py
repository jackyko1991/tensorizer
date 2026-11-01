r"""Mathematics → tensor translation.

1. ζ(s) by Euler–Maclaurin (valid for every s ≠ 1, not only Re s > 1), vectorised over many s_t at once:

    ζ(s_t) = E_{tn} 1_n + N^{1−s_t}/(s_t − 1) + ½ N^{−s_t} + Σ_k B_{2k}/(2k)! · s_t(s_t+1)⋯(s_t+2k−2) · N^{−s_t−2k+1}

   with E_{tn} = n^{−s_t} for n = 1 … N−1 (n summed) and B_{2k} the Bernoulli numbers.

2. On the critical line s = ½ + it the Riemann–Siegel function Z(t) = e^{iθ(t)} ζ(½ + it) is real, so its zeros are
   sign changes. θ(t) = arg Γ(¼ + it/2) − (t/2) ln π has the asymptotic series used below.

3. Primes from zeros. Riemann's prime-power count J(x) = Σ_{p^m ≤ x} 1/m obeys the explicit formula

    J(x) = li(x) − Σ_ρ li(x^ρ) − ln 2 + ∫_x^∞ dt / (t (t² − 1) ln t),

   summed over the zeros ρ = ½ + iγ in conjugate pairs, li(x^ρ) := Ei(ρ ln x). Möbius inversion gives
   π(x) = Σ_n μ(n)/n · J(x^{1/n}), a finite sum because J(y) = 0 for y < 2.
"""

import math

import torch

EULER_GAMMA = 0.5772156649015329
BERNOULLI = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510, 43867 / 798, -174611 / 330]


def zeta(s: torch.Tensor, n_terms: int) -> torch.Tensor:
    r"""ζ(s_t) by Euler–Maclaurin with N = n_terms and 10 Bernoulli corrections.   s: (T,) complex128 → (T,)"""
    n = torch.arange(1, n_terms, dtype=torch.float64, device=s.device)  # 1 … N−1
    head = torch.einsum("tn->t", torch.exp(-s[:, None] * torch.log(n)[None, :]))  # E_{tn} 1_n
    big = torch.tensor(float(n_terms), dtype=torch.float64, device=s.device)
    n_pow = torch.exp(-s * torch.log(big))  # N^{−s}
    total = head + big * n_pow / (s - 1) + 0.5 * n_pow
    rising = s.clone()  # s(s+1)⋯(s+2k−2)
    for k, b in enumerate(BERNOULLI, start=1):
        total = total + b / math.factorial(2 * k) * rising * n_pow / big ** (2 * k - 1)
        rising = rising * (s + 2 * k - 1) * (s + 2 * k)
    return total


def theta(t: torch.Tensor) -> torch.Tensor:
    r"""Riemann–Siegel θ(t) = t/2 ln(t/2π) − t/2 − π/8 + 1/(48t) + 7/(5760t³) + …   (accurate to < 1e-8 for t > 10)."""
    return t / 2 * torch.log(t / (2 * math.pi)) - t / 2 - math.pi / 8 + 1 / (48 * t) + 7 / (5760 * t**3)


def hardy_z(t: torch.Tensor, n_terms: int) -> torch.Tensor:
    r"""Z(t) = Re[e^{iθ(t)} ζ(½ + it)] (the imaginary part vanishes up to rounding).   (T,) → (T,)"""
    return (torch.exp(1j * theta(t)) * zeta(0.5 + 1j * t, n_terms)).real


def riemann_siegel_z(t: torch.Tensor) -> torch.Tensor:
    r"""Riemann–Siegel formula: main sum Z ≈ 2 Σ_{n≤N} n^{−½} cos(θ − t ln n) plus the first correction term
    (−1)^{N−1} (t/2π)^{−¼} C₀(p), N = ⌊√(t/2π)⌋, p = √(t/2π) − N, C₀(p) = cos 2π(p² − p − 1/16) / cos 2πp."""
    a = torch.sqrt(t / (2 * math.pi))
    big_n = a.floor()
    n = torch.arange(1, int(big_n.max()) + 1, dtype=torch.float64, device=t.device)  # (N_max,)
    keep = n[None, :] <= big_n[:, None]  # only n ≤ N(t) counts
    phase = theta(t)[:, None] - t[:, None] * torch.log(n)[None, :]  # (T, N_max)
    main = 2 * torch.einsum("tn->t", torch.where(keep, torch.cos(phase) / torch.sqrt(n), 0.0))
    p = a - big_n
    den = torch.cos(2 * math.pi * p)
    p = torch.where(den.abs() < 1e-9, p + 1e-9, p)  # C₀ has removable 0/0 points at p = ¼, ¾
    c0 = torch.cos(2 * math.pi * (p * p - p - 1 / 16)) / torch.cos(2 * math.pi * p)
    sign = 1 - 2 * ((big_n - 1) % 2)  # (−1)^{N−1}
    return main + sign * a.rsqrt() * c0  # a^{−1/2} = (t/2π)^{−1/4}


def ei(w: torch.Tensor) -> torch.Tensor:
    r"""Exponential integral Ei(w) for complex w (principal branch, cut along the negative real axis).

    |w| < 24: power series Ei(w) = γ + ln w + Σ_k w^k / (k·k!).
    |w| ≥ 24: asymptotic series Ei(w) ≈ e^w/w Σ_{k<24} k!/w^k + iπ·sign(Im w), error below ~1e-9 relative.
    """
    w = w.to(torch.complex128)
    small = w.abs() < 24
    ws = torch.where(small, w, 1.0)
    term, series = ws.clone(), ws.clone()
    for k in range(2, 100):
        term = term * ws * (k - 1) / k**2  # w^k / (k·k!)
        series = series + term
    near = EULER_GAMMA + torch.log(ws) + series
    wl = torch.where(small, 30.0, w)
    term, tail = torch.ones_like(wl), torch.ones_like(wl)
    for k in range(1, 24):
        term = term * k / wl
        tail = tail + term
    far = torch.exp(wl) / wl * tail + 1j * math.pi * torch.sign(wl.imag)
    return torch.where(small, near, far)


def li(x: torch.Tensor) -> torch.Tensor:
    r"""Logarithmic integral li(x) = ∫₀^x dt/ln t (principal value) = Ei(ln x), for real x > 1."""
    return ei(torch.log(x.to(torch.float64))).real


def tail_integral(y: torch.Tensor) -> torch.Tensor:
    r"""∫_y^∞ dt / (t (t² − 1) ln t) = ∫_{ln y}^∞ du / ((e^{2u} − 1) u), trapezoid rule; < 0.15 for y ≥ 2."""
    u = torch.log(y)[:, None] + torch.linspace(0, 12, 3000, dtype=torch.float64, device=y.device)[None, :]  # (Y, U)
    return torch.trapezoid(1 / ((torch.exp(2 * u) - 1) * u), u, dim=1)


def zero_terms(x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    r"""li(x^ρ) + li(x^ρ̄) = 2 Re Ei(ρ ln x) for every zero ρ_k = ½ + iγ_k and every x.   (X,), (K,) → (K, X)"""
    rho = 0.5 + 1j * gammas.to(torch.float64)
    return 2 * ei(rho[:, None] * torch.log(x)[None, :]).real


def gram_r(x: torch.Tensor, zeta_int: torch.Tensor) -> torch.Tensor:
    r"""Riemann's R(x) = Σ_n μ(n)/n li(x^{1/n}) via the Gram series 1 + Σ_k (ln x)^k / (k·k!·ζ(k+1)).
    zeta_int[k] = ζ(k+1) for k = 1 … K.   (X,) → (X,)"""
    lx = torch.log(x.to(torch.float64))
    total, power = torch.ones_like(lx), torch.ones_like(lx)
    for k in range(1, len(zeta_int)):
        power = power * lx / k  # (ln x)^k / k!
        total = total + power / (k * zeta_int[k])
    return total
