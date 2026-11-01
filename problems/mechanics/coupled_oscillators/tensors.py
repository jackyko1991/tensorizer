"""Build the concrete tensors: the stiffness tensor K_ij and the initial displacement x_i(0)."""

import math

import torch


def stiffness(n: int, k: float, device: torch.device | None = None) -> torch.Tensor:
    """K_ij of shape (N, N) for the chain  wall – m – m – … – m – wall.

    Mass i is pulled by the springs on both sides (2k on the diagonal) and
    coupled to its neighbours i ± 1 (−k on the off-diagonals):

        K = k [[ 2, -1,  0, …],
               [-1,  2, -1, …],
               [ 0, -1,  2, …], …]
    """
    diagonal = 2 * k * torch.ones(n, device=device)  # (N,)
    off = -k * torch.ones(n - 1, device=device)  # (N-1,)
    return torch.diag(diagonal) + torch.diag(off, 1) + torch.diag(off, -1)


def analytic_frequencies(n: int, k: float, m: float) -> torch.Tensor:
    """Textbook result ω_n = 2 √(k/m) sin(nπ / 2(N+1)) for n = 1 … N: shape (N,)."""
    modes = torch.arange(1, n + 1)
    return 2 * math.sqrt(k / m) * torch.sin(modes * math.pi / (2 * (n + 1)))


def initial_displacement(n: int, start: str, amplitude: float, device: torch.device | None = None) -> torch.Tensor:
    """x_i(0) of shape (N,). The masses always start at rest, v_i(0) = 0."""
    if start == "first mass":
        x0 = torch.zeros(n, device=device)
        x0[0] = amplitude
    elif start == "middle mass":
        x0 = torch.zeros(n, device=device)
        x0[n // 2] = amplitude
    else:  # one standing wave  sin(p π i / (N+1)),  i = 1 … N
        p = {"mode 1": 1, "mode 2": 2, "highest mode": n}[start]
        i = torch.arange(1, n + 1, device=device)
        x0 = amplitude * torch.sin(p * math.pi * i / (n + 1))
    return x0
