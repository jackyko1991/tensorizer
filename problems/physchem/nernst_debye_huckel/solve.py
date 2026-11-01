"""Solve the linearised Poisson–Boltzmann equation for every concentration at once with torch.linalg.solve."""

import torch

from .equation import pb_system


def solve_pb(r: torch.Tensor, h: torch.Tensor, kap: torch.Tensor, a: float) -> torch.Tensor:
    """φ_ci (C, M) on the radial grids r (C, M): solve A_cij u_cj = b_ci for each c, then φ = u / r.

    torch.linalg.solve treats the leading dimension as a batch: one call, one dense system per concentration.
    """
    A, b = pb_system(h, kap, a, r.shape[1])
    u = torch.linalg.solve(A, b)  # (C, M)
    return u / r
