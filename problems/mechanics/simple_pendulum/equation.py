r"""Physics → tensor translation.

A point mass on a light rod of length L swings under gravity g. Newton's law along the arc gives

    \ddot\theta = -\frac{g}{L} \sin\theta .

We release B pendulums from different amplitudes and keep them in one tensor, so the
batch index b labels the pendulum:

    \ddot\theta_b = -\frac{g}{L} \sin\theta_b

Nothing is summed here: every pendulum obeys the same law independently, and PyTorch
applies sin to all B angles at once. As a first-order system with state y = (θ, ω):

    \dot\theta_b = \omega_b,    \dot\omega_b = -\frac{g}{L} \sin\theta_b

For small angles sin θ ≈ θ and the solution is θ_b(t) = θ_b(0) cos(Ω t) with Ω = √(g/L).
"""

import math

import torch


def angular_acceleration(theta: torch.Tensor, g: float, L: float) -> torch.Tensor:
    r"""\ddot\theta_b = -(g/L) \sin\theta_b.   theta: (B,) → (B,)"""
    return -(g / L) * torch.sin(theta)


def rhs(y: torch.Tensor, g: float, L: float) -> torch.Tensor:
    """dy/dt for the state y of shape (B, 2): y[:, 0] = angle θ, y[:, 1] = angular velocity ω."""
    theta_dot = y[:, 1]
    omega_dot = angular_acceleration(y[:, 0], g, L)
    return torch.stack([theta_dot, omega_dot], dim=1)


def small_angle(theta0: torch.Tensor, t: torch.Tensor, g: float, L: float) -> torch.Tensor:
    r"""Linearised solution θ_{tb} = θ_b(0) cos(Ω t_t).   theta0: (B,), t: (T,) → (T, B)"""
    omega = math.sqrt(g / L)
    return torch.einsum("t,b->tb", torch.cos(omega * t), theta0)  # outer product, nothing summed


def energy(y: torch.Tensor, g: float, L: float) -> torch.Tensor:
    r"""Energy per unit mass E_b = ½ L² ω_b² + g L (1 − cos θ_b).   (..., B, 2) → (..., B)"""
    theta, omega = y[..., 0], y[..., 1]
    return 0.5 * L**2 * omega**2 + g * L * (1 - torch.cos(theta))


def exact_period(theta0: torch.Tensor, g: float, L: float) -> torch.Tensor:
    r"""T_b = 4 √(L/g) K(sin(θ_b/2)), with the complete elliptic integral K from the
    arithmetic–geometric mean:  K(k) = π / (2 AGM(1, √(1 − k²))).   (B,) → (B,)"""
    k = torch.sin(theta0 / 2)
    a, b = torch.ones_like(k), torch.sqrt(1 - k**2)
    for _ in range(10):  # the AGM converges quadratically: 10 rounds is plenty
        a, b = (a + b) / 2, torch.sqrt(a * b)
    return 4 * math.sqrt(L / g) * math.pi / (2 * a)
