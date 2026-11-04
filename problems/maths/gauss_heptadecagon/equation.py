r"""Mathematics → tensor translation: Gauss periods, the nested radical, ruler-and-compass geometry, Gauss–Wantzel.

1. Gauss periods. With a primitive root g mod 17 the exponents are ordered e_j = g^j mod 17, j = 0 … 15. At level m
   (m = 0 … 4) the 2^m periods of length 16 / 2^m are

        η_{m,i} = Σ_j M_{mij} ζ^{e_j},      M_{mij} = [j ≡ i mod 2^m]      (summed index: j)

   so a whole level is one masked sum over the cyclic order, and all five levels are one einsum.

2. The quadratics. Period i of level m splits into the two periods i and i + 2^m of level m + 1, with
        sum      η_{m+1,i} + η_{m+1,i+2^m} = η_{m,i}
        product  η_{m+1,i} η_{m+1,i+2^m}   = c_i + Σ_k C_{ik} η_{m,k}       (summed index: k)
   The product is a sum of ζ^{a+b} over the exponents a, b of the two halves; counting how often each a + b mod 17
   occurs gives integers that are constant on the level-m periods (that is Gauss's key lemma). So every η of level
   m + 1 is a root of t² − s t + p = 0 with s, p known from level m: one square root per level.

3. Ruler and compass. A construction step draws a line through two points or a circle about a point through
   another; new points are line–line, line–circle and circle–circle intersections, written below as tensor ops.

4. Gauss–Wantzel. The regular n-gon is constructible iff n = 2^k p₁ ⋯ p_r with distinct Fermat primes p_i, iff
   Euler's φ(n) is a power of 2.
"""

import torch

from .tensors import FERMAT_PRIMES, P, level_masks

# ---------------------------------------------------------------- 1–2. periods and their quadratics


def periods(zeta: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """η_{m,i} = Σ_j M_{mij} ζ^{e_j} for all five levels at once.   (17,), (16,) → (5, 16) complex, padded with 0"""
    masks = level_masks(zeta.device).to(zeta.dtype)  # (5, 16, 16)
    return torch.einsum("mij,j->mi", masks, zeta[order.to(zeta.device)])


def product_in_periods(order: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact integer coefficients of η_{m+1,i} η_{m+1,i+2^m} = c_i + Σ_k C_{ik} η_{m,k}.   → (2^m,), (2^m, 2^m)"""
    n = 2**m
    exps = order.reshape(-1, 2 * n).T  # (2n, L): row r = exponents of period r at level m + 1
    a, b = exps[:n], exps[n:]  # the two halves of every level-m period
    sums = (a[:, :, None] + b[:, None, :]) % P  # (n, L, L) exponent of each term ζ^a ζ^b
    counts = torch.nn.functional.one_hot(sums, P).sum(dim=(1, 2))  # (n, 17): how often each exponent occurs
    on_periods = counts[:, order.reshape(-1, n)]  # (n, 16/n, n): count at every exponent of every level-m period
    if not (on_periods == on_periods[:, :1]).all():
        raise ArithmeticError("product is not a combination of level-m periods")
    return counts[:, 0], on_periods[:, 0]


def solve_quadratic(s: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both roots of t² − s t + p = 0, with one complex square root: t = (s ± √(s² − 4p)) / 2."""
    root = torch.sqrt(s * s - 4 * p)
    return (s + root) / 2, (s - root) / 2


def gauss_formula(dtype: torch.dtype) -> torch.Tensor:
    """Gauss (1796): 16 cos(2π/17) = −1 + √17 + √(34 − 2√17) + 2√(17 + 3√17 − √(34 − 2√17) − 2√(34 + 2√17))."""
    s17 = torch.sqrt(torch.tensor(17.0, dtype=dtype))
    a = torch.sqrt(34 - 2 * s17)
    b = torch.sqrt(34 + 2 * s17)
    return (-1 + s17 + a + 2 * torch.sqrt(17 + 3 * s17 - a - 2 * b)) / 16


# ---------------------------------------------------------------- 3. ruler and compass


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def line_line(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Intersection of line ab with line cd: a + t(b − a) with t = (c − a) × (d − c) / (b − a) × (d − c).   → (2,)"""
    t = _cross(c - a, d - c) / _cross(b - a, d - c)
    return a + t * (b - a)


def line_circle(a: torch.Tensor, b: torch.Tensor, centre: torch.Tensor, r) -> torch.Tensor:
    """Both points a + t(b − a) at distance r from the centre: |f + t d|² = r² with d = b − a, f = a − centre.

    Rows sorted by t (first the one nearer a's side).   → (2, 2)"""
    d, f = b - a, a - centre
    qa, qb, qc = d @ d, 2 * (f @ d), f @ f - r**2
    root = torch.sqrt(qb * qb - 4 * qa * qc)
    t = torch.stack([(-qb - root), (-qb + root)]) / (2 * qa)  # (2,)
    return a + t[:, None] * d


def circle_circle(c1: torch.Tensor, r1, c2: torch.Tensor, r2) -> torch.Tensor:
    """Both intersections of two circles: along the centre line at distance a from c1, then ± h across it.

    Row 0 lies to the left of the direction c1 → c2, row 1 to the right.   → (2, 2)"""
    d = (c2 - c1).norm()
    e = (c2 - c1) / d  # unit vector c1 → c2
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = torch.sqrt(r1**2 - a**2)
    perp = torch.stack([-e[1], e[0]])  # e rotated by +90°
    return c1 + a * e + torch.stack([h, -h])[:, None] * perp


# ---------------------------------------------------------------- 4. Gauss–Wantzel


def euler_phi(n_max: int, device: torch.device | None = None) -> torch.Tensor:
    """φ(n) for n = 0 … n_max by a sieve: for every prime p, φ(n) ← φ(n)(1 − 1/p) on its multiples.   → (n_max+1,)"""
    phi = torch.arange(n_max + 1, device=device)
    for p in range(2, n_max + 1):
        if phi[p] == p:  # untouched so far ⇒ p is prime
            phi[p::p] -= phi[p::p] // p
    return phi


def is_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """x > 0 with a single bit set: x & (x − 1) = 0."""
    return (x > 0) & ((x & (x - 1)) == 0)


def fermat_factorisation(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """n = 2^k · (odd part). The n-gon is constructible iff the odd part divides 3·5·17·257·65537 = 2^32 − 1, i.e. it
    is a product of distinct Fermat primes.   (N,) → 2^k (N,), Fermat primes dividing n (N, 5), constructible (N,)"""
    two_power = n & -n  # lowest set bit
    odd = n // two_power
    fermat = torch.tensor(FERMAT_PRIMES, device=n.device)
    divides = odd[:, None] % fermat[None, :] == 0  # (N, 5)
    constructible = ((2**32 - 1) % odd == 0) & (n >= 3)
    return two_power, divides, constructible
