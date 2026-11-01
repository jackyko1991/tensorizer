r"""Physics → tensor translation.

Near a black hole of mass M the pull of gravity changes quickly with distance. Two nearby particles at
x and x + ξ feel slightly different accelerations; to first order in the separation ξ (Taylor expansion)

    \ddot ξ_i = T_{ij}\, ξ_j,        T_{ij} = -\partial_i \partial_j Φ      (j summed)

T_{ij} is the tidal tensor. For a point mass, Φ = -GM/r, and

    T_{ij} = \frac{GM}{r^5}\,\left(3 x_i x_j - r^2 δ_{ij}\right)

Its trace vanishes outside the mass (∇²Φ = 0), and its eigenvalues are 2GM/r³ along the radial direction
(stretching) and −GM/r³ twice across it (squeezing): a falling body becomes a long thin noodle —
"spaghettification". Here T_{ij} is computed both from this formula and as the Hessian of Φ by automatic
differentiation (torch.func), which works for any potential.

A body held together by its own gravity (mass m, radius R) is torn apart when the tidal stretch 2GMR/r³
beats its surface gravity Gm/R², i.e. inside the tidal (Roche) radius

    r_t = R\,(2M/m)^{1/3}.
"""

import torch
from torch.func import grad, jacrev, vmap


def potential(x: torch.Tensor, GM: float) -> torch.Tensor:
    r"""Φ = -GM / \sqrt{x_j x_j}.   x: (..., 3) → (...)"""
    return -GM / torch.sqrt(torch.einsum("...j,...j->...", x, x))


def tidal_tensor(x: torch.Tensor, GM: float) -> torch.Tensor:
    r"""T_{ij} = GM (3 x_i x_j - r^2 δ_{ij}) / r^5.   x: (..., 3) → (..., 3, 3)"""
    r2 = torch.einsum("...j,...j->...", x, x)[..., None, None]  # (..., 1, 1)
    outer = torch.einsum("...i,...j->...ij", x, x)  # (..., 3, 3)
    delta = torch.eye(3, dtype=x.dtype, device=x.device)
    return GM * (3 * outer - r2 * delta) / r2**2.5


def tidal_tensor_autograd(x: torch.Tensor, GM: float) -> torch.Tensor:
    """T_ij = -∂_i∂_j Φ as the Hessian of the potential, by automatic differentiation.   (P, 3) → (P, 3, 3)"""
    return -vmap(jacrev(grad(lambda p: potential(p, GM))))(x)


def acceleration(x: torch.Tensor, GM: float) -> torch.Tensor:
    r"""Exact Newtonian pull \ddot x_{bi} = -GM (x_{bj} x_{bj})^{-3/2} x_{bi}.   (B, 3) → (B, 3)"""
    r2 = torch.einsum("bj,bj->b", x, x)
    return -GM * torch.einsum("b,bi->bi", r2**-1.5, x)


def tidal_acceleration(T: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    r"""\ddot ξ_{bi} = T_{ij} ξ_{bj} (j summed): linear tidal law for every ring particle b.   (3, 3), (B, 3) → (B, 3)"""
    return torch.einsum("ij,bj->bi", T, xi)


def rhs(y: torch.Tensor, GM: float, n: int) -> torch.Tensor:
    """dy/dt for y of shape (1 + 2n, 2, 3): [centre of the ring, n ring particles (exact), n separations ξ (tidal law)]."""
    x, v = y[:, 0], y[:, 1]
    a_exact = acceleration(x[: 1 + n], GM)  # centre and ring particles, full gravity
    a_tidal = tidal_acceleration(tidal_tensor(x[0], GM), x[1 + n :])  # separations from the centre, linearised
    return torch.stack([v, torch.cat([a_exact, a_tidal])], dim=1)


def tidal_radius(R_body: float, mass_ratio: float) -> float:
    """r_t = R (2M/m)^{1/3}, mass_ratio = m/M."""
    return R_body * (2 / mass_ratio) ** (1 / 3)
