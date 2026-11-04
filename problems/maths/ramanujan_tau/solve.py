"""Expand Δ(q) = q Π (1 − qⁿ)²⁴: (η³)² = η⁶, (η⁶)² = η¹², (η¹²)² = η²⁴, then shift by one power of q."""

from collections.abc import Callable

import torch

from . import tensors
from .equation import digits_to_float, garner_balanced, mul_exact, mul_mod


def _delta(n_max: int, seed: torch.Tensor, mul: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """τ(0), …, τ(n_max) from the series of Π (1 − qⁿ)³ (length n_max) and a truncated product."""
    e = seed
    for _ in range(3):
        e = mul(e, e)  # exponent 3 → 6 → 12 → 24
    return torch.cat(
        [e.new_zeros(*e.shape[:-1], 1), e], dim=-1
    )  # the factor q: τ(n) is the coefficient of q^{n−1} in Π (1 − qⁿ)²⁴


def tau_exact(n_max: int, device: torch.device) -> torch.Tensor:
    """Exact τ(n) in int64 by Toeplitz products (fine up to n ≈ 1000).   → (N + 1,) int64"""
    return _delta(n_max, tensors.eta_cubed(n_max, device, torch.int64), mul_exact)


def tau_mod(n_max: int, moduli: list[int], device: torch.device) -> torch.Tensor:
    """Exact residues of τ(n) modulo several m at once, one row each, by FFT products with rounding.
    → (M, N + 1) int64"""
    m = torch.tensor(moduli, device=device)[:, None]  # (M, 1)
    seed = tensors.eta_cubed(n_max, device, torch.int64)[None, :] % m  # (M, N)
    return _delta(n_max, seed, lambda a, b: mul_mod(a, b, m))


def tau_float(n_max: int, device: torch.device) -> torch.Tensor:
    """τ(n) for large n: exact residues modulo a batch of primes, then the Chinese remainder theorem, read out in
    float64 (so every value is correct to float64 rounding, even where τ(n) ≈ 10^30).   → (N + 1,) float64"""
    moduli = tensors.crt_moduli(n_max, device)
    residues = tau_mod(n_max, moduli, device)  # (M, N + 1)
    return digits_to_float(garner_balanced(residues.T, moduli), moduli)
