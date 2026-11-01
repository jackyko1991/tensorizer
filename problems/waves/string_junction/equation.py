r"""Physics → tensor translation.

A string under tension T whose mass per length μ(x) jumps at the knot x = 0:

    \mu(x) \, \partial_t^2 u = T \, \partial_x^2 u

On the grid the second derivative is a stencil applied at every point,

    \mu_i \ddot u_i = T D_{ij} u_j,        D_{ij} u_j = (u_{i+1} - 2 u_i + u_{i-1}) / \Delta x^2    (j summed)

which is exactly a 1D convolution with the kernel (1, −2, 1). Matching u and T u_x at the knot gives

    R = (Z_1 - Z_2) / (Z_1 + Z_2),     \mathcal T = 2 Z_1 / (Z_1 + Z_2),     Z = \sqrt{T \mu}.
"""

import math

import torch
import torch.nn.functional as F


def second_derivative(u: torch.Tensor, dx: float) -> torch.Tensor:
    r"""D_{ij} u_j by convolution with (1, −2, 1) / Δx². u: (N,) → (N,), zero at the two fixed ends."""
    kernel = torch.tensor([1.0, -2.0, 1.0], dtype=u.dtype, device=u.device).view(1, 1, 3)
    inner = F.conv1d(u.view(1, 1, -1), kernel).view(-1) / dx**2  # (N − 2,) interior points only
    return F.pad(inner, (1, 1))  # the fixed ends do not move


def acceleration(u: torch.Tensor, mu: torch.Tensor, T: float, dx: float) -> torch.Tensor:
    r"""\ddot u_i = (T / μ_i) D_{ij} u_j.   (N,) → (N,)"""
    return T / mu * second_derivative(u, dx)


def impedance(T: float, mu: float) -> float:
    """Z = √(T μ)."""
    return math.sqrt(T * mu)


def reflection_transmission(Z1: float, Z2: float) -> tuple[float, float]:
    """Amplitude ratios R = (Z1 − Z2)/(Z1 + Z2) and 𝒯 = 2 Z1/(Z1 + Z2)."""
    return (Z1 - Z2) / (Z1 + Z2), 2 * Z1 / (Z1 + Z2)


def energy(traj: torch.Tensor, mu: torch.Tensor, T: float, dx: float, dt: float) -> torch.Tensor:
    r"""E = Σ_i ½ μ_i \dot u_i² Δx + ½ T Σ (Δu/Δx)² Δx at every time.

    traj: (S, N). The velocity is the centred difference (u^{n+1} − u^{n−1}) / 2Δt, so the result is (S − 2,).
    """
    v = (traj[2:] - traj[:-2]) / (2 * dt)  # (S-2, N)
    ux = (traj[1:-1, 1:] - traj[1:-1, :-1]) / dx  # (S-2, N-1)
    return 0.5 * torch.einsum("i,si->s", mu, v**2) * dx + 0.5 * T * torch.einsum("si,si->s", ux, ux) * dx
