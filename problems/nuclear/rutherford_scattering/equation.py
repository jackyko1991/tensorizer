r"""Physics → tensor translation.

An alpha particle (mass m) moving past a heavy nucleus feels the repulsive Coulomb force

    m \ddot{\mathbf r} = k \, \mathbf r / r^3,        k = Z_α Z e^2 / 4π ε_0 = 2 Z · 1.44 MeV fm.

For a batch of particles b in the scattering plane (components i, j ∈ {x, y}):

    \ddot x_{bi} = (k/m) \, x_{bi} / (x_{bj} x_{bj})^{3/2}          (j summed),

with the conserved energy and angular momentum

    E_b = ½ m v_{bi} v_{bi} + k / r_b,        L_b = m \, ε_{ij} x_{bi} v_{bj}       (i, j summed).

Energy E and impact parameter b fix the scattering angle and the Rutherford cross-section:

    \cot(θ/2) = 2 E b / k,        dσ/dΩ = (k / 4E)^2 / \sin^4(θ/2).
"""

import torch


def acceleration(x: torch.Tensor, k: float, m: float) -> torch.Tensor:
    r"""\ddot x_{bi} = (k/m) x_{bi} / r_b^3,  r_b^2 = x_{bj} x_{bj}.   x: (B, 2) → (B, 2)"""
    r2 = torch.einsum("bj,bj->b", x, x)  # (B,)
    return (k / m) * x / r2[:, None] ** 1.5


def rhs(y: torch.Tensor, k: float, m: float) -> torch.Tensor:
    """dy/dt for y of shape (B, 2, 2): y[:, 0] = position (fm), y[:, 1] = velocity (units of c)."""
    return torch.stack([y[:, 1], acceleration(y[:, 0], k, m)], dim=1)


def energy(y: torch.Tensor, k: float, m: float) -> torch.Tensor:
    r"""E_b = ½ m v_{bi} v_{bi} + k / r_b  (MeV).   (..., 2, 2) → (...)"""
    x, v = y[..., 0, :], y[..., 1, :]
    r = torch.einsum("...i,...i->...", x, x).sqrt()
    return 0.5 * m * torch.einsum("...i,...i->...", v, v) + k / r


def angular_momentum(y: torch.Tensor, m: float) -> torch.Tensor:
    r"""L_b = m ε_{ij} x_{bi} v_{bj}  (MeV fm / c).   (..., 2, 2) → (...)"""
    eps = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=y.dtype, device=y.device)  # 2D Levi-Civita ε_ij
    return m * torch.einsum("ij,...i,...j->...", eps, y[..., 0, :], y[..., 1, :])


def scattering_angle(v: torch.Tensor) -> torch.Tensor:
    r"""θ_b = angle between the outgoing velocity and the beam axis x.   v: (B, 2) → (B,) in [0, π]"""
    speed = torch.einsum("bi,bi->b", v, v).sqrt()
    return torch.arccos((v[:, 0] / speed).clamp(-1.0, 1.0))


def rutherford_angle(b: torch.Tensor, energy_mev: float, k: float) -> torch.Tensor:
    r"""θ(b) = 2 \arctan(k / 2Eb)."""
    return 2 * torch.atan(k / (2 * energy_mev * b))


def rutherford_cross_section(theta: torch.Tensor, energy_mev: float, k: float) -> torch.Tensor:
    r"""dσ/dΩ = (k/4E)^2 / \sin^4(θ/2)  in fm²/sr."""
    return (k / (4 * energy_mev)) ** 2 / torch.sin(theta / 2) ** 4


def closest_approach(b: torch.Tensor, energy_mev: float, k: float) -> torch.Tensor:
    r"""r_{min} = (d/2)(1 + \sqrt{1 + (2b/d)^2}),  d = k/E the head-on distance."""
    d = k / energy_mev
    return 0.5 * d * (1 + torch.sqrt(1 + (2 * b / d) ** 2))
