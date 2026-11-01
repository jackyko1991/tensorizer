"""Diagonalise all L radial Hamiltonians in one batched eigh, then pick out the orbitals (n, l)."""

import math

import torch


def eigenstates(H: torch.Tensor, dr: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (E_{ln} (L, N), u_{lin} (L, N, N)) with ∫ u² dr = 1 and u > 0 just outside the nucleus.

    eigh works on the last two axes and loops over the leading batch axis l by itself.
    """
    energies, u = torch.linalg.eigh(H)  # (L, N), (L, N, N)
    u = u / math.sqrt(dr)
    sign = torch.sign(u[:, :1, :])  # (L, 1, N) sign of each state at the first grid point
    return energies, u * torch.where(sign == 0, 1.0, sign)


def orbitals(n_max: int) -> list[tuple[int, int]]:
    """All (n, l) with n ≤ n_max, in the order 1s, 2s, 2p, 3s, 3p, 3d, …"""
    return [(n, ell) for n in range(1, n_max + 1) for ell in range(n)]


def pick(energies: torch.Tensor, u: torch.Tensor, n: int, ell: int) -> tuple[float, torch.Tensor]:
    """Energy and u(r) of orbital (n, l): column n − l − 1 of block l (radial quantum number)."""
    column = n - ell - 1
    return energies[ell, column].item(), u[ell, :, column]


def count_nodes(u: torch.Tensor, tol: float = 1e-4) -> int:
    """Sign changes of u(r), ignoring the tiny noisy tail where |u| < tol · max|u|."""
    big = u[u.abs() > tol * u.abs().max()]
    return int((big[1:] * big[:-1] < 0).sum())


def radial_on_slice(u: torch.Tensor, dr: float, r: torch.Tensor) -> torch.Tensor:
    """R = u/r evaluated at arbitrary radii r (any shape) by linear interpolation of the grid values."""
    u0 = torch.cat([u.new_zeros(1), u])  # (N+1,) add u(0) = 0, so u0[i] sits at r = i·dr
    position = (r / dr).clamp(0, len(u) - 0.001)  # fractional index
    i0 = position.floor().long()
    w = position - i0
    u_r = (1 - w) * u0[i0] + w * u0[i0 + 1]
    return u_r / r
