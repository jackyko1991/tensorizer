"""Run Euler's recurrence for p(n), exactly (int64), approximately (float64) or modulo several m at once."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import digits_to_float, euler_step, garner_digits


def partitions(
    n_max: int, device: torch.device, dtype: torch.dtype, moduli: torch.Tensor | None = None
) -> torch.Tensor:
    """p(0), …, p(n_max) by the pentagonal recurrence.   → (N + 1,) or, with moduli (M,), residues (N + 1, M)

    int64 is exact up to tensors.N_EXACT; float64 keeps ~13 significant digits much further; residues mod m are
    exact for every n because only numbers below m² · √n ever appear."""
    g, s = tensors.pentagonal(n_max, device)
    count = torch.searchsorted(g, torch.arange(n_max + 1, device=device), right=True).tolist()  # g_j ≤ n
    shape = (n_max + 1,) if moduli is None else (n_max + 1, len(moduli))
    p = torch.zeros(shape, dtype=dtype, device=device)
    p[0] = 1
    for i in steps(n_max, "p(n)"):
        n = i + 1
        j = count[n]
        p[n] = euler_step(p, n, g[:j], s[:j], moduli)
    return p


def partitions_float(n_max: int, device: torch.device) -> torch.Tensor:
    """p(0), …, p(n_max) to full float64 precision: exact residues modulo primes near 2^31, then the Chinese
    remainder theorem. (The float64 recurrence itself loses digits, see the notes.)   → (N + 1,) float64"""
    primes = tensors.crt_primes(n_max, device)
    residues = partitions(n_max, device, torch.int64, primes)  # (N + 1, M) exact
    return digits_to_float(garner_digits(residues, primes), primes)
