r"""Mathematics → tensor translation: Ramanujan's τ(n) as the coefficients of a power series.

1. The discriminant modular form

        Δ(q) = q Π_{n≥1} (1 − qⁿ)²⁴ = Σ_{n≥1} τ(n) qⁿ.

   Starting from Jacobi's sparse series for Π (1 − qⁿ)³, three squarings give the 6th, 12th and 24th powers. A
   truncated product c_n = Σ_j a_j b_{n−j} (repeated j ⇒ summed) is done three ways: as a Toeplitz matrix times a
   vector in int64 (exact, small n), by FFT in float64 (large n) and by FFT followed by rounding modulo m (exact
   residues).

2. Hecke theory (Mordell 1917): τ(mn) = τ(m) τ(n) for gcd(m, n) = 1 and
   τ(p^{k+1}) = τ(p) τ(p^k) − p¹¹ τ(p^{k−1}).

3. Ramanujan's congruence τ(n) ≡ σ₁₁(n) (mod 691), σ₁₁(n) = Σ_{d|n} d¹¹.

4. Deligne (1974): |τ(p)| ≤ 2 p^{11/2}, so x_p = τ(p) / (2 p^{11/2}) ∈ [−1, 1]. Sato–Tate (proved for Δ in 2011):
   the x_p are distributed with density (2/π) √(1 − x²), whose even moments are the Catalan numbers C_k / 4^k.
"""

import math

import torch


def mul_exact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Truncated product c_n = Σ_{j≤n} a_{n−j} b_j as a Toeplitz matrix times a vector, int64.   (L,), (L,) → (L,)"""
    n = len(a)
    idx = torch.arange(n, device=a.device)
    lag = idx[:, None] - idx[None, :]  # (L, L): n − j
    toeplitz = torch.where(lag >= 0, a[lag.clamp(min=0)], 0)
    return toeplitz @ b


def mul_float(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The same truncated product by FFT along the last axis (zero-padded against wrap-around), float64.
    (..., L), (..., L) → (..., L)"""
    n = a.shape[-1]
    spec = torch.fft.rfft(a.double(), 2 * n) * torch.fft.rfft(b.double(), 2 * n)
    return torch.fft.irfft(spec, 2 * n)[..., :n]


def mul_mod(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor | int) -> torch.Tensor:
    """Truncated products of residues, one row per modulus: FFT, round to the exact integer, reduce.
    (M, L), (M, L), m (M, 1) → (M, L) int64

    Exact while the true sums (< m² L) stay far below 2^53, where float64 still counts in steps of one; the moduli
    are chosen with m² L ≤ 2^42, which leaves the FFT rounding error near 10^−2."""
    return mul_float(a, b).round().long() % m


def garner_balanced(residues: torch.Tensor, m: list[int]) -> torch.Tensor:
    """Chinese remainder theorem for every n at once, with balanced digits −m_i/2 < v_i ≤ m_i/2 so that negative
    numbers come out negative:  x = v_0 + v_1 m_0 + v_2 m_0 m_1 + …   (N, M) → (N, M) int64"""
    v = residues.clone()
    v[:, 0] = torch.where(v[:, 0] > m[0] // 2, v[:, 0] - m[0], v[:, 0])
    for i in range(1, len(m)):
        for j in range(i):
            v[:, i] = (v[:, i] - v[:, j]) % m[i] * pow(m[j], -1, m[i]) % m[i]
        v[:, i] = torch.where(v[:, i] > m[i] // 2, v[:, i] - m[i], v[:, i])
    return v


def digits_to_float(v: torch.Tensor, m: list[int]) -> torch.Tensor:
    """x = v_0 + m_0 (v_1 + m_1 (v_2 + …)) by Horner's rule in float64.   (N, M) → (N,)"""
    x = v[:, -1].double()
    for i in range(v.shape[1] - 2, -1, -1):
        x = v[:, i].double() + m[i] * x
    return x


def multiplicativity(tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every coprime pair 1 < m < n with mn ≤ N, and whether τ(mn) = τ(m) τ(n).   (N + 1,) → (P,), (P,), (P,) bool"""
    n_max = len(tau) - 1
    r = torch.arange(2, n_max // 2 + 1, device=tau.device)
    m, n = torch.meshgrid(r, r, indexing="ij")
    keep = (m < n) & (m * n <= n_max) & (torch.gcd(m, n) == 1)
    m, n = m[keep], n[keep]
    return m, n, tau[m * n] == tau[m] * tau[n]


def hecke(tau: torch.Tensor, primes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both sides of τ(p^{k+1}) = τ(p) τ(p^k) − p¹¹ τ(p^{k−1}) for every prime power p^{k+1} ≤ N.
    (N + 1,) int64, (P,) → lhs (H,), rhs (H,)"""
    n_max = len(tau) - 1
    k = torch.arange(1, 64, device=tau.device)
    p, k = torch.meshgrid(primes.double(), k.double(), indexing="ij")
    keep = (k + 1) * torch.log(p) <= math.log(n_max) + 1e-9  # p^{k+1} ≤ N, decided before any power can overflow
    p, k = p[keep].long(), k[keep].long()
    p, k = p[p ** (k + 1) <= n_max], k[p ** (k + 1) <= n_max]
    lhs = tau[p ** (k + 1)]
    rhs = tau[p] * tau[p**k] - p**11 * tau[p ** (k - 1)]
    return lhs, rhs


def sigma_mod(n_max: int, k: int, mod: int, device: torch.device) -> torch.Tensor:
    """σ_k(n) = Σ_{d|n} d^k mod `mod` for n = 0 … n_max (σ_k(0) := 0).   → (N + 1,) int64

    Every pair (d, j) with d j ≤ N is listed once (about N ln N pairs) and d^k is added onto n = d j."""
    d_all = torch.arange(1, n_max + 1, device=device)
    power = torch.ones_like(d_all)
    for _ in range(k):
        power = power * d_all % mod  # d^k mod `mod`
    count = n_max // d_all  # multiples of d up to N
    d = torch.repeat_interleave(d_all, count)
    start = torch.cumsum(count, 0) - count
    j = torch.arange(len(d), device=device) - torch.repeat_interleave(start, count) + 1
    sigma = torch.zeros(n_max + 1, dtype=torch.int64, device=device)
    sigma.index_add_(0, d * j, power[d - 1])
    return sigma % mod


def normalise(tau_p: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """x_p = τ(p) / (2 p^{11/2}), in [−1, 1] by Deligne's theorem.   (P,) → (P,)"""
    return tau_p.double() / (2 * p.double() ** 5.5)


def semicircle(x: torch.Tensor) -> torch.Tensor:
    """Sato–Tate density (2/π) √(1 − x²) on [−1, 1]."""
    return 2 / math.pi * torch.sqrt((1 - x**2).clamp(min=0))


def semicircle_moments(k_max: int) -> torch.Tensor:
    """E[x^{2k}] = C_k / 4^k for k = 1 … k_max (C_k the Catalan numbers).   → (K,)"""
    return torch.tensor([math.comb(2 * k, k) / (k + 1) / 4**k for k in range(1, k_max + 1)], dtype=torch.float64)
