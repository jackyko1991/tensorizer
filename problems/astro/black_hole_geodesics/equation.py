r"""Physics → tensor translation.

General relativity: a freely falling particle follows a geodesic of spacetime,

    \frac{d^2 x^\mu}{d\tau^2} = -\Gamma^\mu{}_{\alpha\beta}\,\dot x^\alpha \dot x^\beta      (α, β summed)

where τ is the particle's proper time and the Christoffel symbols come from the metric g_{μν}(x):

    \Gamma^\mu{}_{\alpha\beta} = ½\, g^{\mu\sigma}\left(\partial_\alpha g_{\sigma\beta}
                                 + \partial_\beta g_{\sigma\alpha} - \partial_\sigma g_{\alpha\beta}\right)   σ summed

Outside a non-rotating black hole of mass M (units G = c = 1, coordinates x^μ = (t, r, θ, φ), signature −+++)
the metric is Schwarzschild's:

    g_{\mu\nu} = \operatorname{diag}\left(-(1 - 2M/r),\; (1 - 2M/r)^{-1},\; r^2,\; r^2\sin^2\theta\right)

The horizon is at r_s = 2M. We let torch.func.jacrev differentiate g_{μν} (no hand-derived Christoffels) and
contract with einsum. For a massive particle the 4-velocity stays normalised, g_{μν} \dot x^μ \dot x^ν = −1.

Landmarks of the effective potential (per unit mass, L = angular momentum)

    ½\dot r^2 + V(r) = ½(E^2 - 1),     V = -\frac{M}{r} + \frac{L^2}{2r^2} - \frac{M L^2}{r^3}

(the last term is what Newton lacks): the innermost stable circular orbit at r = 6M, the photon sphere at
r = 3M, and a bound orbit's perihelion advancing by ≈ 6πM / (a(1 − e²)) per turn.
"""

import torch
from torch.func import jacrev, vmap


def metric(x: torch.Tensor, M: float) -> torch.Tensor:
    r"""g_{μν}(x) for one event x = (t, r, θ, φ): (4,) → (4, 4)."""
    r, theta = x[1], x[2]
    f = 1 - 2 * M / r
    return torch.diag(torch.stack([-f, 1 / f, r**2, (r * torch.sin(theta)) ** 2]))


def christoffel(x: torch.Tensor, M: float) -> torch.Tensor:
    r"""Γ^μ_{αβ} at a batch of events by automatic differentiation.   x: (B, 4) → (B, 4, 4, 4)

    dg[z, σ, β, α] = ∂_α g_{σβ}  (z = batch index; jacrev appends the derivative index last)
    """
    g = vmap(lambda p: metric(p, M))(x)  # (B, 4, 4)
    g_inv = torch.linalg.inv(g)  # (B, 4, 4)  g^{μσ}
    dg = vmap(jacrev(lambda p: metric(p, M)))(x)  # (B, 4, 4, 4)
    d1 = dg.permute(0, 1, 3, 2)  # [z, σ, α, β] = ∂_α g_{σβ}
    d2 = dg  # [z, σ, α, β] = ∂_β g_{σα}
    d3 = dg.permute(0, 3, 1, 2)  # [z, σ, α, β] = ∂_σ g_{αβ}
    return 0.5 * torch.einsum("zms,zsab->zmab", g_inv, d1 + d2 - d3)


def geodesic_acceleration(x: torch.Tensor, u: torch.Tensor, M: float) -> torch.Tensor:
    r"""\ddot x^μ = -Γ^μ_{αβ} \dot x^α \dot x^β (α, β summed).   x, u: (B, 4) → (B, 4)"""
    return -torch.einsum("zmab,za,zb->zm", christoffel(x, M), u, u)  # z = batch index


def rhs(y: torch.Tensor, M: float) -> torch.Tensor:
    """dy/dτ for y of shape (B, 2, 4): y[:, 0] = x^μ, y[:, 1] = dx^μ/dτ."""
    return torch.stack([y[:, 1], geodesic_acceleration(y[:, 0], y[:, 1], M)], dim=1)


def norm(y: torch.Tensor, M: float) -> torch.Tensor:
    r"""g_{μν} \dot x^μ \dot x^ν (= −1 for a massive particle).   y: (..., 2, 4) → (...)"""
    flat = y.reshape(-1, 2, 4)
    g = vmap(lambda p: metric(p, M))(flat[:, 0])
    return torch.einsum("zmn,zm,zn->z", g, flat[:, 1], flat[:, 1]).reshape(y.shape[:-2])


def newton_acceleration(x: torch.Tensor, M: float) -> torch.Tensor:
    r"""For comparison, Newton: \ddot x_{bi} = -M (x_{bj} x_{bj})^{-3/2} x_{bi}.   (B, 2) → (B, 2)"""
    r2 = torch.einsum("bj,bj->b", x, x)
    return -M * torch.einsum("b,bi->bi", r2**-1.5, x)


def newton_rhs(y: torch.Tensor, M: float) -> torch.Tensor:
    """dy/dt for a Newtonian orbit, y of shape (B, 2, 2) = (position, velocity) in the orbital plane."""
    return torch.stack([y[:, 1], newton_acceleration(y[:, 0], M)], dim=1)


def effective_potential(r: torch.Tensor, L: float, M: float, relativistic: bool = True) -> torch.Tensor:
    """V(r) per unit mass; relativistic=False drops the −ML²/r³ term (Newton)."""
    V = -M / r + L**2 / (2 * r**2)
    return V - M * L**2 / r**3 if relativistic else V


def isco_radius(M: float) -> float:
    """Innermost stable circular orbit, 6M (= 3 r_s)."""
    return 6 * M


def photon_sphere_radius(M: float) -> float:
    """Circular orbit of light, 3M."""
    return 3 * M


def weak_field_precession(p: float, M: float) -> float:
    """Perihelion advance per orbit 6πM/p, with p = a(1 − e²) the semi-latus rectum."""
    return 6 * torch.pi * M / p
