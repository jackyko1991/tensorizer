r"""Physics → tensor translation.

The viscous Burgers equation is the simplest model of a fluid that carries itself along:

    u_t + u u_x = ν u_xx      ⇔      u_t = -(½ u^2)_x + ν u_xx

On a periodic grid x_i the derivatives become matrices: (∂_x f)_i = D_{ij} f_j and
(∂_x^2 f)_i = L_{ij} f_j, so the PDE turns into a system of ODEs (method of lines)

    \dot u_i = -½ D_{ij} (u_j)^2 + ν L_{ij} u_j

The repeated index j is summed: each is one einsum('ij,j->i'). The square (u_j)^2 is elementwise.
"""

import torch


def rhs(u: torch.Tensor, D: torch.Tensor, L: torch.Tensor, nu: float) -> torch.Tensor:
    r"""\dot u_i = -½ D_{ij} u_j^2 + ν L_{ij} u_j   (j summed).   u: (N,), D, L: (N, N) → (N,)"""
    flux = 0.5 * u**2  # (N,) the "momentum flux" ½u², squared elementwise
    return -torch.einsum("ij,j->i", D, flux) + nu * torch.einsum("ij,j->i", L, u)


def slope(u: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    r"""u_x as (∂_x u)_i = D_{ij} u_j for one or many snapshots.   u: (..., N) → (..., N)"""
    return torch.einsum("ij,...j->...i", D, u)


def mass(u: torch.Tensor, dx: float) -> torch.Tensor:
    r"""∫ u dx ≈ Σ_i u_i Δx — conserved on a periodic domain because the PDE is a pure divergence."""
    return u.sum(dim=-1) * dx
