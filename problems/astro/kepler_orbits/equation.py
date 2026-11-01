r"""Physics → tensor translation.

Newton's law of gravity for a planet around a star of mass M (the star is much heavier, so it stays put):

    \ddot{\mathbf r} = -\frac{GM}{r^3}\,\mathbf r

In index notation for a batch of planets b and components i ∈ {x, y}:

    \ddot x_{bi} = -GM \,(x_{bj} x_{bj})^{-3/2}\, x_{bi}

A repeated index (j above) is summed — that is einsum('bj,bj->b'). Two quantities stay constant:

    energy per unit mass            E_b = ½ v_{bi} v_{bi} - GM / \sqrt{x_{bj} x_{bj}}
    angular momentum per unit mass  L_b = \varepsilon_{ij} x_{bi} v_{bj}          (i and j summed)

L_b / 2 is the area swept per unit time (Kepler's second law), and the semi-major axis a_b = -GM / (2 E_b)
fixes the period T_b = 2π \sqrt{a_b^3 / GM} (Kepler's third law).
"""

import torch

EPS2 = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])  # ε_ij, the 2D Levi-Civita symbol


def acceleration(x: torch.Tensor, GM: float) -> torch.Tensor:
    r"""\ddot x_{bi} = -GM (x_{bj} x_{bj})^{-3/2} x_{bi}.   x: (B, 2) → (B, 2)"""
    r2 = torch.einsum("bj,bj->b", x, x)  # (B,)  squared distance to the star
    return -GM * torch.einsum("b,bi->bi", r2**-1.5, x)


def rhs(y: torch.Tensor, GM: float) -> torch.Tensor:
    """dy/dt for the state y of shape (B, 2, 2): y[:, 0] = position, y[:, 1] = velocity."""
    return torch.stack([y[:, 1], acceleration(y[:, 0], GM)], dim=1)


def energy(y: torch.Tensor, GM: float) -> torch.Tensor:
    r"""E_b = ½ v_{bi} v_{bi} - GM / r_b.   y: (..., 2, 2) → (...)"""
    x, v = y[..., 0, :], y[..., 1, :]
    return 0.5 * torch.einsum("...i,...i->...", v, v) - GM / torch.einsum("...i,...i->...", x, x).sqrt()


def angular_momentum(y: torch.Tensor) -> torch.Tensor:
    r"""L_b = ε_{ij} x_{bi} v_{bj} (= x v_y - y v_x).   y: (..., 2, 2) → (...)"""
    x, v = y[..., 0, :], y[..., 1, :]
    return torch.einsum("ij,...i,...j->...", EPS2.to(y), x, v)


def areal_velocity(y: torch.Tensor) -> torch.Tensor:
    """dA/dt = L / 2: the area swept per unit time by the line star → planet."""
    return 0.5 * angular_momentum(y)


def semi_major_axis(y: torch.Tensor, GM: float) -> torch.Tensor:
    """a = -GM / (2E) for a bound orbit."""
    return -GM / (2 * energy(y, GM))


def kepler_period(a: torch.Tensor, GM: float) -> torch.Tensor:
    r"""Kepler's third law T = 2π \sqrt{a^3 / GM}."""
    return 2 * torch.pi * torch.sqrt(a**3 / GM)
