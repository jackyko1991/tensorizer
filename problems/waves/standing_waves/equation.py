r"""Physics → tensor translation.

The wave equation for a string with fixed ends,

    \partial_t^2 u = c^2 \partial_x^2 u,       u(0, t) = u(L, t) = 0,

is discretised on M interior grid points. The second derivative becomes a matrix D_{ij} (the stencil
1, −2, 1 divided by Δx²), and B strings are moved together:

    \ddot u_{bi} = c^2 D_{ij} u_{bj}        (j is repeated ⇒ summed)

Time is discretised with the centred difference (leap-frog):

    u^{n+1}_{bi} = 2 u^n_{bi} - u^{n-1}_{bi} + \Delta t^2 \ddot u^n_{bi}

The eigenvalues λ_n of D give the normal modes: ω_n² = −c² λ_n, close to (nπc/L)².
"""

import math

import torch


def acceleration(u: torch.Tensor, D: torch.Tensor, c: float) -> torch.Tensor:
    r"""\ddot u_{bi} = c^2 D_{ij} u_{bj}   — j summed.   u: (B, M), D: (M, M) → (B, M)"""
    return c**2 * torch.einsum("ij,bj->bi", D, u)


def leapfrog_step(u: torch.Tensor, u_prev: torch.Tensor, D: torch.Tensor, c: float, dt: float) -> torch.Tensor:
    """u^{n+1} = 2 u^n − u^{n−1} + Δt² ü^n.   All tensors (B, M)."""
    return 2 * u - u_prev + dt**2 * acceleration(u, D, c)


def mode_frequencies(D: torch.Tensor, c: float) -> torch.Tensor:
    """Frequencies f_n = ω_n / 2π of the discrete string, lowest first: (M,)."""
    lam = torch.linalg.eigvalsh(D)  # all negative
    omega = c * torch.sqrt(-lam)
    return torch.sort(omega).values / (2 * math.pi)


def energy(traj: torch.Tensor, dx: float, dt: float, c: float) -> torch.Tensor:
    r"""Energy per unit mass density  E_b = ½ Σ_i (\dot u_{bi}² + c² (∂_x u)_{bi}²) Δx.

    traj: (T, B, N) including the fixed ends. The velocity uses centred differences in time,
    so the result has T − 2 rows: (T − 2, B).
    """
    v = (traj[2:] - traj[:-2]) / (2 * dt)  # (T-2, B, N)
    ux = (traj[1:-1, :, 1:] - traj[1:-1, :, :-1]) / dx  # (T-2, B, N-1) slope between grid points
    kinetic = 0.5 * torch.einsum("tbi,tbi->tb", v, v) * dx
    potential = 0.5 * c**2 * torch.einsum("tbi,tbi->tb", ux, ux) * dx
    return kinetic + potential
