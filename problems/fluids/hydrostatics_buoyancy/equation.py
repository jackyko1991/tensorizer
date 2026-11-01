r"""Physics → tensor translation.

Hydrostatic pressure: the pressure grows with depth because the water above has weight,

    \nabla p = \rho_w \mathbf g    ⇒    p(\mathbf x) = p_0 + \rho_w g_i x_i

with g_i = (0, -g) and the water surface at z = 0. The repeated index i is summed (einsum).

Each block b (a cube of side s and density ρ_b) feels its weight, the buoyant force
(Archimedes: the weight of the displaced water) and a damping drag while in the water:

    \rho_b s^3 \ddot z_b = \rho_w g s^2 d_b - \rho_b g s^3 - \gamma \rho_w s^2 d_b\,\dot z_b,   d_b = clamp(-z_b, 0, s)

where z_b is the height of the block's bottom face and d_b its submerged depth. The drag grows with
the wet volume s^2 d_b, like the buoyancy. Dividing by the block mass ρ_b s^3 gives the acceleration
used below; both water forces carry the same factor (ρ_w/ρ_b)(d_b/s).
"""

import torch


def pressure(x: torch.Tensor, g: torch.Tensor, p0: float, rho_w: float) -> torch.Tensor:
    r"""p = p_0 + \rho_w g_i x_i  (i summed).   x: (..., 2) points → (...) pressures in Pa."""
    return p0 + rho_w * torch.einsum("i,...i->...", g, x)


def submerged_depth(z: torch.Tensor, s: float) -> torch.Tensor:
    """How much of each block is under the surface: d_b = clamp(-z_b, 0, s).   z: (B,) → (B,)"""
    return torch.clamp(-z, 0.0, s)


def acceleration(
    z: torch.Tensor, v: torch.Tensor, rho_b: torch.Tensor, s: float, g: float, rho_w: float, gamma: float
) -> torch.Tensor:
    r"""\ddot z_b = g (\rho_w d_b / (\rho_b s) - 1) - \gamma (\rho_w d_b / (\rho_b s)) \dot z_b.   All (B,).

    Buoyancy and drag both act on the wet part of the block only, so both carry the factor ρ_w d_b / (ρ_b s).
    """
    wet = rho_w / rho_b * submerged_depth(z, s) / s  # (B,) displaced water mass / block mass
    return g * (wet - 1.0) - gamma * wet * v


def rhs(y: torch.Tensor, rho_b: torch.Tensor, s: float, g: float, rho_w: float, gamma: float) -> torch.Tensor:
    """dy/dt for the state y of shape (B, 2): y[:, 0] = height z_b of the bottom face, y[:, 1] = velocity."""
    z, v = y[:, 0], y[:, 1]
    return torch.stack([v, acceleration(z, v, rho_b, s, g, rho_w, gamma)], dim=1)
