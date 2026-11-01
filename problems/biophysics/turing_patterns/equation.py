r"""Physics → tensor translation: Gray–Scott reaction–diffusion and its linear stability.

Indices: b parameter pair (F_b, k_b), (i, j) grid cell, q wavenumber.

    ∂u/∂t = D_u ∇²u - u v² + F (1 - u)          u: substrate ("food"), fed in at rate F
    ∂v/∂t = D_v ∇²v + u v² - (F + k) v          v: autocatalyst, removed at rate F + k

On the grid (spacing h, periodic) the Laplacian is the 5-point stencil
    (∇²u)_{bij} = (u_{b,i+1,j} + u_{b,i-1,j} + u_{b,i,j+1} + u_{b,i,j-1} - 4 u_{bij}) / h².
No index is summed in the update: every cell of every parameter pair advances on its own, coupled only to its
four neighbours by the stencil.

Linear stability of a uniform state (u*, v*): small waves δ ∝ exp(i q·x + λ t) grow at the eigenvalues of
    M(q) = J - q² diag(D_u, D_v),   J = [[-v*² - F, -2u*v*], [v*², 2u*v* - (F + k)]].
Turing instability: stable without diffusion (Re λ(0) < 0) but Re λ(q) > 0 for a band of q ≠ 0.
"""

import math

import torch


def laplacian(u: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """5-point periodic Laplacian of u (B, n, n): neighbour sum minus four times the centre, over h².   (B, n, n)"""
    neighbours = u.roll(1, -2) + u.roll(-1, -2) + u.roll(1, -1) + u.roll(-1, -1)
    return (neighbours - 4 * u) / h**2


def reaction(u: torch.Tensor, v: torch.Tensor, F: torch.Tensor, k: torch.Tensor):
    """Local kinetics R_u = -u v² + F (1 - u),  R_v = u v² - (F + k) v.   F, k: (B, 1, 1) broadcast over the grid."""
    uvv = u * v * v  # the autocatalytic step U + 2V → 3V
    return -uvv + F * (1 - u), uvv - (F + k) * v


def rhs(y: torch.Tensor, F: torch.Tensor, k: torch.Tensor, Du: float, Dv: float) -> torch.Tensor:
    """dy/dt for y = (u, v) of shape (2, B, n, n)."""
    u, v = y[0], y[1]
    Ru, Rv = reaction(u, v, F, k)
    return torch.stack([Du * laplacian(u) + Ru, Dv * laplacian(v) + Rv])


def uniform_states(F: float, k: float) -> list[tuple[float, float]]:
    """Uniform steady states (u*, v*): always (1, 0), and where F ≥ 4 (F + k)² two more,
    v* = [F ± √(F² - 4F(F + k)²)] / 2(F + k),  u* = (F + k) / v*   (from u v = F + k and F (1 - u) = u v²)."""
    states = [(1.0, 0.0)]
    disc = F * F - 4 * F * (F + k) ** 2
    if disc >= 0:
        for sign in (1.0, -1.0):
            v = (F + sign * math.sqrt(disc)) / (2 * (F + k))
            states.append(((F + k) / v, v))
    return states


def jacobian(u: float, v: float, F: float, k: float) -> torch.Tensor:
    """J = ∂(R_u, R_v)/∂(u, v) at a uniform state.   (2, 2)"""
    return torch.tensor([[-v * v - F, -2 * u * v], [v * v, 2 * u * v - (F + k)]], dtype=torch.float64)


def growth_rate(J: torch.Tensor, Du: float, Dv: float, q: torch.Tensor) -> torch.Tensor:
    """Dispersion relation: largest Re λ of M(q) = J - q² diag(D_u, D_v) for every wavenumber q (Q,).   (Q,)"""
    D = torch.diag(torch.tensor([Du, Dv], dtype=J.dtype))
    M = J - q.to(J.dtype)[:, None, None] ** 2 * D  # (Q, 2, 2)
    return torch.linalg.eigvals(M).real.max(-1).values


def lattice_q2(q: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """What the 5-point stencil does to a wave along one axis: -∇² exp(iqx) = (4/h²) sin²(qh/2) exp(iqx)."""
    return 4 / h**2 * torch.sin(q * h / 2) ** 2
