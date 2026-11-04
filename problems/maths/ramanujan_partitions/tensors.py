"""Build the concrete tensors: the generalised pentagonal numbers, the moduli and Ramanujan's congruences."""

import math

import torch

N_EXACT = 405  # p(405) = 9 147 679 068 859 117 602 < 2^63 ≈ 9.22·10^18 < p(406): the int64 limit


def pentagonal(n_max: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised pentagonal numbers g = k(3k − 1)/2 for k = 1, −1, 2, −2, … up to n_max, with Euler's signs.

    Π_k (1 − q^k) = Σ_k (−1)^k q^{k(3k−1)/2}, so p(n) = Σ_j s_j p(n − g_j) with s = +, +, −, −, +, +, …
    → g (G,) int64 sorted, s (G,) int64
    """
    k_max = math.isqrt(2 * n_max // 3 + 1) + 2
    k = torch.arange(1, k_max + 1, device=device)
    g = torch.stack([k * (3 * k - 1) // 2, k * (3 * k + 1) // 2], dim=1).flatten()  # 1, 2, 5, 7, 12, 15, …
    s = torch.where(k % 2 == 1, 1, -1).repeat_interleave(2)
    keep = g <= n_max
    return g[keep], s[keep]


def crt_primes(n_max: int, device: torch.device) -> torch.Tensor:
    """The largest primes below 2^31, enough of them that their product exceeds p(n_max).   → (M,) int64

    log₂ p(n) < π√(2n/3) / ln 2, and each prime adds about 31 bits. The recurrence only adds and subtracts residues,
    so the sums stay below 2^31 · (number of pentagonal terms) and never leave int64."""
    bits = math.pi * math.sqrt(2 * n_max / 3) / math.log(2)
    count = math.ceil((bits + 2) / 30) + 1
    sieve = torch.ones(46341, dtype=torch.bool, device=device)  # trial divisors up to √(2^31)
    sieve[:2] = False
    for d in range(2, 216):  # 216² > 46341
        if sieve[d]:
            sieve[d * d :: d] = False
    small = torch.nonzero(sieve).flatten()
    cand = torch.arange(2**31 - 1, 2**31 - 1 - 200 * count, -2, device=device)  # odd numbers just below 2^31
    is_prime = (cand[:, None] % small[None, :] != 0).all(dim=1)
    return cand[is_prime][:count]


def moduli(device: torch.device) -> torch.Tensor:
    """The moduli of Ramanujan's congruences, one column each in the residue table.   → (M,) int64"""
    return torch.tensor([5, 7, 11, 25, 49, 121], device=device)


def congruences(device: torch.device) -> torch.Tensor:
    """Rows (a, b, m) with p(a k + b) ≡ 0 (mod m) for every k ≥ 0, in the order of moduli().   → (M, 3) int64

    Ramanujan proved 5, 7 and 25 (1919) and 11 and 49 in a manuscript published after his death (1921); Watson
    (1938) proved all powers of 5 and 7, Atkin (1967) all powers of 11, including 121. In each case 24 b ≡ 1 (mod m)."""
    return torch.tensor([[5, 4, 5], [7, 5, 7], [11, 6, 11], [25, 24, 25], [49, 47, 49], [121, 116, 121]], device=device)
