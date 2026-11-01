r"""Physics → tensor translation.

The Navier–Cauchy equation: Newton's law for a homogeneous isotropic elastic solid,

    ρ \ddot u_i = (λ + μ) ∂_i ∂_j u_j + μ ∂_j ∂_j u_i + f_i        (j summed)

On the grid we build the table of second derivatives H_{ijk} = ∂_i ∂_j u_k at every node
(central differences), then both terms are contractions of H:

    ∂_i ∂_j u_j = einsum('ijj...->i...', H)      (gradient of the divergence → P waves)
    ∂_j ∂_j u_i = einsum('jji...->i...', H)      (Laplacian of each component → S waves)

Taking the divergence or curl of the equation splits it into two wave equations with speeds

    c_P = \sqrt{(λ + 2μ)/ρ}  for θ = ∂_j u_j,        c_S = \sqrt{μ/ρ}  for ω = ε_{ij} ∂_i u_j.
"""

import math

import torch


def d1(f: torch.Tensor, axis: int, h: float) -> torch.Tensor:
    """Central first derivative along grid axis 0 (x) or 1 (y) of a field (..., Nx, Ny)."""
    dim = f.dim() - 2 + axis
    return (torch.roll(f, -1, dim) - torch.roll(f, 1, dim)) / (2 * h)


def hessian(u: torch.Tensor, h: float) -> torch.Tensor:
    """H_{ijk} = ∂_i ∂_j u_k by central differences.   u: (2, Nx, Ny) → (2, 2, 2, Nx, Ny)."""
    dxx = (torch.roll(u, -1, 1) - 2 * u + torch.roll(u, 1, 1)) / h**2
    dyy = (torch.roll(u, -1, 2) - 2 * u + torch.roll(u, 1, 2)) / h**2
    dxy = d1(d1(u, 0, h), 1, h)
    return torch.stack([torch.stack([dxx, dxy]), torch.stack([dxy, dyy])])  # [i, j, k, x, y]


def acceleration(u: torch.Tensor, h: float, lam: float, mu: float, rho: float) -> torch.Tensor:
    r"""\ddot u_i = [(λ+μ) ∂_i∂_j u_j + μ ∂_j∂_j u_i] / ρ.   (2, Nx, Ny) → (2, Nx, Ny)"""
    H = hessian(u, h)
    grad_div = torch.einsum("ijj...->i...", H)  # ∂_i ∂_j u_j
    laplace = torch.einsum("jji...->i...", H)  # ∂_j ∂_j u_i
    return ((lam + mu) * grad_div + mu * laplace) / rho


def divergence(u: torch.Tensor, h: float) -> torch.Tensor:
    """θ = ∂_j u_j (volume change, carried by P waves): (2, Nx, Ny) → (Nx, Ny)."""
    return d1(u[0], 0, h) + d1(u[1], 1, h)


def curl(u: torch.Tensor, h: float) -> torch.Tensor:
    """ω = ∂_x u_y - ∂_y u_x (rotation, carried by S waves): (2, Nx, Ny) → (Nx, Ny)."""
    return d1(u[1], 0, h) - d1(u[0], 1, h)


def wave_speeds(lam: float, mu: float, rho: float) -> tuple[float, float]:
    """c_P = sqrt((λ + 2μ)/ρ), c_S = sqrt(μ/ρ)."""
    return math.sqrt((lam + 2 * mu) / rho), math.sqrt(mu / rho)


def stable_time_step(h: float, c_p: float, c_s: float, courant: float) -> float:
    """Δt = S h / sqrt(c_P² + c_S²). The explicit scheme is only stable for S ≤ 1, so larger S is refused."""
    if courant > 1.0:
        raise ValueError(f"Courant number {courant} > 1: the leapfrog scheme would blow up")
    return courant * h / math.sqrt(c_p**2 + c_s**2)
