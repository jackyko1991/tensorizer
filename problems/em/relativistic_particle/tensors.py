"""Build the concrete tensors: metric, Levi-Civita symbol, fields and the particle batch."""

import itertools

import torch

DTYPE = torch.float64  # U^0 = γ grows large under constant E: keep the extra digits
EVEN = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]  # cyclic permutations of (x, y, z)


def metric() -> torch.Tensor:
    """η_{μν} = diag(+1, −1, −1, −1) of shape (4, 4)."""
    return torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=DTYPE))


def levi_civita() -> torch.Tensor:
    """ε_ijk of shape (3, 3, 3): +1 for even permutations, −1 for odd ones, 0 when an index repeats."""
    eps = torch.zeros(3, 3, 3, dtype=DTYPE)
    for i, j, k in itertools.permutations(range(3)):
        eps[i, j, k] = 1.0 if (i, j, k) in EVEN else -1.0
    return eps


def fields(config: str, strength: float) -> tuple[torch.Tensor, torch.Tensor]:
    """E_i and B_i, shape (3,) each: B along z for "cyclotron", E along x for "accelerate"."""
    E, B = torch.zeros(3, dtype=DTYPE), torch.zeros(3, dtype=DTYPE)
    if config == "cyclotron":
        B[2] = strength
    else:
        E[0] = strength
    return E, B


def particles(config: str, n: int, v_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (q/m (P,), initial state y0 (P, 2, 4)) with every particle starting at the origin at t = 0.

    cyclotron:  q/m = 1, speeds v_p from 0.3 up to v_max along +y, so U^μ = γ (1, 0, v, 0)
    accelerate: at rest, U^μ = (1, 0, 0, 0), with q/m = 0.25, 0.5, 1, 2, ... (doubling)
    """
    y0 = torch.zeros(n, 2, 4, dtype=DTYPE)
    if config == "cyclotron":
        v = torch.linspace(0.3, v_max, n, dtype=DTYPE) if n > 1 else torch.tensor([v_max], dtype=DTYPE)
        gamma = 1 / torch.sqrt(1 - v**2)
        y0[:, 1, 0] = gamma  # U^0 = γ
        y0[:, 1, 2] = gamma * v  # U^y = γ v
        qm = torch.ones(n, dtype=DTYPE)
    else:
        y0[:, 1, 0] = 1.0  # at rest: U^μ = (1, 0, 0, 0)
        qm = 0.25 * 2.0 ** torch.arange(n, dtype=DTYPE)
    return qm, y0
