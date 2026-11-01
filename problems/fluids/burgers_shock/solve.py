"""March the method-of-lines ODEs with RK4, and evaluate the exact Cole–Hopf solution for comparison."""

import math

import torch

from problems._common.integrators import integrate

from .equation import rhs


def time_step(dx: float, u_max: float, nu: float) -> float:
    """Stable RK4 step: limited by advection (Δx/|u|) and by diffusion (Δx²/ν)."""
    return min(0.5 * dx / u_max, 0.25 * dx**2 / nu)


def simulate(
    u0: torch.Tensor, D: torch.Tensor, L: torch.Tensor, nu: float, dt: float, t_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), u (T, N))."""
    n_steps = int(math.ceil(t_max / dt))
    u = integrate(lambda u, t: rhs(u, D, L, nu), u0, dt, n_steps)
    times = torch.arange(n_steps + 1, device=u0.device, dtype=u0.dtype) * dt
    return times, u


def cole_hopf(x: torch.Tensor, t: float, nu: float, amplitude: float, offset: float, n_y: int = 6000) -> torch.Tensor:
    r"""Exact solution of viscous Burgers for u_0 = U_0 + A sin x (Cole–Hopf transform).

    u = -2ν ∂_x ln φ turns Burgers into the heat equation φ_t = ν φ_xx, whose solution is a Gaussian
    average. The result is a weighted mean over starting points y:

        u(x, t) = Σ_y w(x, y) (x - y) / t,     w = softmax_y[ -(x - y)² / (4νt) - F(y) / (2ν) ]

    with F(y) = ∫_0^y u_0 = U_0 y + A (1 - cos y). softmax keeps the huge exponents finite.
    """
    if t <= 0:
        return offset + amplitude * torch.sin(x)
    y = torch.linspace(-3 * math.pi, 5 * math.pi, n_y, dtype=torch.float64, device=x.device)  # (M,)
    F = offset * y + amplitude * (1 - torch.cos(y))  # (M,)
    gap = x[:, None] - y[None, :]  # (N, M)
    w = torch.softmax(-(gap**2) / (4 * nu * t) - F / (2 * nu), dim=1)  # (N, M), rows sum to 1
    return torch.einsum("nm,nm->n", w, gap) / t
