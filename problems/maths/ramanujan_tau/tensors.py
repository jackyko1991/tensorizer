"""Build the concrete tensors: Jacobi's sparse series for Π (1 − qⁿ)³ and the primes."""

import math

import torch


def eta_cubed(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Coefficients of Π_{n≥1} (1 − qⁿ)³ up to q^{length − 1} from Jacobi's identity:

        Π (1 − qⁿ)³ = Σ_{k≥0} (−1)^k (2k + 1) q^{k(k+1)/2},

    so only about √(2·length) coefficients are nonzero.   → (length,)"""
    k = torch.arange(0, math.isqrt(2 * length) + 2, device=device)
    pos = k * (k + 1) // 2  # triangular numbers 0, 1, 3, 6, 10, …
    val = torch.where(k % 2 == 0, 1, -1) * (2 * k + 1)
    keep = pos < length
    series = torch.zeros(length, dtype=dtype, device=device)
    series[pos[keep]] = val[keep].to(dtype)
    return series


def primes_up_to(n: int, device: torch.device) -> torch.Tensor:
    """All primes ≤ n by the sieve of Eratosthenes.   → (P,) int64"""
    is_prime = torch.ones(n + 1, dtype=torch.bool, device=device)
    is_prime[:2] = False
    for p in range(2, math.isqrt(n) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return torch.nonzero(is_prime).flatten()


def crt_moduli(n_max: int, device: torch.device) -> list[int]:
    """Primes m with m² · n_max ≤ 2^42 (so FFT products of residues round exactly), enough of them that their product
    exceeds 2 |τ(n)| for every n ≤ n_max. Deligne: |τ(n)| ≤ d(n) n^{11/2}, and the divisor count d(n) ≤ 2√n."""
    cap = math.isqrt(2**42 // n_max)
    bits = 6 * math.log2(n_max) + 4  # log₂(2 · 2√n · n^{11/2}) plus a margin
    count = math.ceil(bits / math.log2(cap)) + 1
    return primes_up_to(cap, device)[-count:].tolist()
