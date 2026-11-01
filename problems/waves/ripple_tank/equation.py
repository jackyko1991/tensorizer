r"""Physics → tensor translation.

The damped, driven 2D wave equation

    \partial_t^2 u + \gamma \partial_t u = c^2 \nabla^2 u + f

with the five-point Laplacian written as a 3 × 3 kernel K sliding over the grid (α, β summed):

    (\nabla^2 u)_{ab} = \frac{1}{h^2} K_{\alpha\beta} u_{a+\alpha, b+\beta}        ← conv2d(u, K, padding=1)

Leap-frog in time with the damping centred, (u^{n+1} − u^{n−1}) / 2Δt:

    (1 + \gamma\Delta t/2) u^{n+1} = 2u^n - (1 - \gamma\Delta t/2) u^{n-1} + \Delta t^2 (c^2 \nabla^2 u^n + f^n)

The explicit scheme is stable only for c Δt / h ≤ 1/√2 (Courant–Friedrichs–Lewy in 2D).
"""

import math

import torch
import torch.nn.functional as F


def laplacian(u: torch.Tensor, K: torch.Tensor, h: float) -> torch.Tensor:
    r"""(∇²u)_{ab} = K_{αβ} u_{a+α, b+β} / h².   u: (n, n), K: (1, 1, 3, 3) → (n, n); zero outside the grid."""
    return F.conv2d(u[None, None], K, padding=1)[0, 0] / h**2


def time_step(h: float, c: float, courant: float) -> float:
    """Δt = S h / c, refusing S > 1/√2 where the explicit 2D scheme blows up."""
    if not 0 < courant <= 1 / math.sqrt(2):
        raise ValueError(f"Courant number {courant} violates the 2D CFL limit 1/sqrt(2) = {1 / math.sqrt(2):.4f}")
    return courant * h / c


def step(u, u_prev, f, gamma, K, c: float, h: float, dt: float) -> torch.Tensor:
    """One damped leap-frog step. All fields (n, n)."""
    a = 0.5 * gamma * dt
    return (2 * u - (1 - a) * u_prev + dt**2 * (c**2 * laplacian(u, K, h) + f)) / (1 + a)


def path_difference(X: torch.Tensor, Y: torch.Tensor, s1: tuple, s2: tuple) -> torch.Tensor:
    """|r1 − r2|: difference of the distances to the two sources at every cell.   (n, n)"""
    r1 = torch.sqrt((X - s1[0]) ** 2 + (Y - s1[1]) ** 2)
    r2 = torch.sqrt((X - s2[0]) ** 2 + (Y - s2[1]) ** 2)
    return (r1 - r2).abs()
