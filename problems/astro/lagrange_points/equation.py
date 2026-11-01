r"""Physics → tensor translation.

Circular restricted three-body problem: a star (mass 1 − μ) and a planet (mass μ) circle their centre of mass,
and a test particle feels both. In the frame rotating with the pair (angular velocity Ω = ẑ) the two masses sit
still at x_1 = (−μ, 0, 0) and x_2 = (1 − μ, 0, 0). Units: G(M_1 + M_2) = 1, separation 1, Ω = 1.

The effective potential adds the centrifugal term to gravity:

    Φ_eff = -\frac{1-μ}{|x - x_1|} - \frac{μ}{|x - x_2|} - ½\left(x_j x_j - (Ω_j x_j)^2\right)

and the equation of motion in the rotating frame has gravity, Coriolis and centrifugal forces,
written with the Levi-Civita symbol ε_{ijk} (all repeated indices summed):

    \ddot x_i = -\sum_s m_s \frac{(x - x_s)_i}{|x - x_s|^3}
                - 2\,ε_{ijk} Ω_j \dot x_k
                - ε_{ijk} Ω_j\, ε_{klm} Ω_l x_m

Lagrange points are where a particle can sit at rest: ∂_i Φ_eff = 0. Newton's method finds them,

    x ← x - (∂_i ∂_j Φ_eff)^{-1}\, ∂_j Φ_eff,

with the gradient and Hessian from torch.func (automatic differentiation). The Jacobi constant
C = -2Φ_eff - \dot x_i \dot x_i is conserved along every orbit.
"""

import torch
from torch.func import grad, jacrev, vmap


def levi_civita(dtype=torch.float64, device=None) -> torch.Tensor:
    """ε_ijk, shape (3, 3, 3)."""
    eps = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[i, j, k], eps[i, k, j] = 1.0, -1.0
    return eps


def primaries(mu: float, dtype=torch.float64, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions x_s (2, 3) and masses m_s (2,) of the star and the planet in the rotating frame."""
    pos = torch.tensor([[-mu, 0.0, 0.0], [1 - mu, 0.0, 0.0]], dtype=dtype, device=device)
    return pos, torch.tensor([1 - mu, mu], dtype=dtype, device=device)


def effective_potential(x: torch.Tensor, mu: float) -> torch.Tensor:
    r"""Φ_eff at points x (..., 3) → (...). Ω = ẑ, so (Ω_j x_j)² = z²."""
    pos, m = primaries(mu, x.dtype, x.device)
    d = x[..., None, :] - pos  # (..., 2, 3) separation from each mass s
    r = torch.sqrt(torch.einsum("...sj,...sj->...s", d, d))  # (..., 2)
    gravity = -torch.einsum("s,...s->...", m, 1 / r)
    centrifugal = -0.5 * (torch.einsum("...j,...j->...", x, x) - x[..., 2] ** 2)
    return gravity + centrifugal


def grad_potential(x: torch.Tensor, mu: float) -> torch.Tensor:
    """∂_i Φ_eff by automatic differentiation, for a batch of points.   (P, 3) → (P, 3)"""
    return vmap(grad(lambda p: effective_potential(p, mu)))(x)


def hessian_potential(x: torch.Tensor, mu: float) -> torch.Tensor:
    """∂_i ∂_j Φ_eff by automatic differentiation.   (P, 3) → (P, 3, 3)"""
    return vmap(jacrev(grad(lambda p: effective_potential(p, mu))))(x)  # reverse mode twice


def acceleration(x: torch.Tensor, v: torch.Tensor, mu: float, eps: torch.Tensor) -> torch.Tensor:
    r"""\ddot x_i = gravity - 2 ε_{ijk} Ω_j v_k - ε_{ijk} Ω_j ε_{klm} Ω_l x_m.   x, v: (B, 3) → (B, 3)"""
    pos, m = primaries(mu, x.dtype, x.device)
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=x.device)
    d = x[:, None, :] - pos  # (B, 2, 3)
    r3 = torch.einsum("bsj,bsj->bs", d, d) ** 1.5  # (B, 2)
    gravity = -torch.einsum("s,bs,bsi->bi", m, 1 / r3, d)
    coriolis = -2 * torch.einsum("ijk,j,bk->bi", eps, omega, v)
    omega_cross_x = torch.einsum("klm,l,bm->bk", eps, omega, x)
    centrifugal = -torch.einsum("ijk,j,bk->bi", eps, omega, omega_cross_x)
    return gravity + coriolis + centrifugal


def rhs(y: torch.Tensor, mu: float, eps: torch.Tensor) -> torch.Tensor:
    """dy/dt for the state y (B, 2, 3): y[:, 0] = position, y[:, 1] = velocity (rotating frame)."""
    return torch.stack([y[:, 1], acceleration(y[:, 0], y[:, 1], mu, eps)], dim=1)


def jacobi_constant(y: torch.Tensor, mu: float) -> torch.Tensor:
    r"""C = -2Φ_eff - \dot x_i \dot x_i.   y: (..., 2, 3) → (...)"""
    x, v = y[..., 0, :], y[..., 1, :]
    return -2 * effective_potential(x, mu) - torch.einsum("...i,...i->...", v, v)
