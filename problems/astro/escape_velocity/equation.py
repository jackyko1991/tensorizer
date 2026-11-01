r"""Physics → tensor translation.

A rocket coasting (engine off) above a planet of mass M only feels the planet's gravity:

    \ddot{\mathbf r} = -\frac{GM}{r^3}\,\mathbf r

For a batch of rockets b and components i ∈ {x, y} (repeated j ⇒ summed):

    \ddot x_{bi} = -GM\,(x_{bj} x_{bj})^{-3/2}\,x_{bi}

The energy per unit mass is conserved,

    E_b = ½ v_{bi} v_{bi} - \frac{GM}{\sqrt{x_{bj} x_{bj}}},

and decides everything: E < 0 → the rocket is bound and falls back, E ≥ 0 → it escapes to infinity.
Setting E = 0 at the surface r = R gives the escape speed v_esc = \sqrt{2GM/R}. A rocket fired
straight up with v < v_esc stops at the height where all its kinetic energy is used up:

    ½ v^2 - GM/R = -GM/r_max   ⇒   r_max = R / (1 - v^2 / v_esc^2)

Michell (1783) and Laplace (1796) asked how small a star must be for v_esc to reach the speed of light c.
Setting \sqrt{2GM/R} = c gives the "dark star" radius R = 2GM/c^2 — exactly the Schwarzschild radius of a
black hole in general relativity (a lucky coincidence of the Newtonian argument).
"""

import math

import torch


def acceleration(x: torch.Tensor, GM: float) -> torch.Tensor:
    r"""\ddot x_{bi} = -GM (x_{bj} x_{bj})^{-3/2} x_{bi}.   x: (B, 2) → (B, 2)"""
    r2 = torch.einsum("bj,bj->b", x, x)  # (B,) squared distance from the planet's centre
    return -GM * torch.einsum("b,bi->bi", r2**-1.5, x)


def rhs(y: torch.Tensor, GM: float) -> torch.Tensor:
    """dy/dt for the state y of shape (B, 2, 2): y[:, 0] = position, y[:, 1] = velocity."""
    return torch.stack([y[:, 1], acceleration(y[:, 0], GM)], dim=1)


def energy(y: torch.Tensor, GM: float) -> torch.Tensor:
    r"""E_b = ½ v_{bi} v_{bi} - GM / r_b.   y: (..., 2, 2) → (...)"""
    x, v = y[..., 0, :], y[..., 1, :]
    return 0.5 * torch.einsum("...i,...i->...", v, v) - GM / torch.einsum("...i,...i->...", x, x).sqrt()


def escape_speed(GM: float, R: float) -> float:
    r"""v_esc = \sqrt{2GM/R}: the launch speed with exactly zero total energy."""
    return math.sqrt(2 * GM / R)


def max_radius(v: torch.Tensor, GM: float, R: float) -> torch.Tensor:
    """Highest point of a vertical launch with v < v_esc, from energy conservation."""
    return R / (1 - v**2 / escape_speed(GM, R) ** 2)


def dark_star_radius(GM: float, c: float) -> float:
    """Radius at which the escape speed equals c: R = 2GM/c² (the Schwarzschild radius)."""
    return 2 * GM / c**2
