r"""Coupled oscillators → tensor translation.

Kuramoto model (Kuramoto 1975). N oscillators with phases θ_i and natural frequencies ω_i, all coupled to all:

    \dot\theta_i = \omega_i + \frac KN \sum_j \sin(\theta_j - \theta_i).

The complex mean field (order parameter) r e^{iψ} = (1/N) Σ_j e^{iθ_j} turns the O(N²) double sum into O(N):

    \dot\theta_i = \omega_i + K\,r\,\sin(\psi - \theta_i)
                 = \omega_i + K\,\operatorname{Im}\!\big(r e^{i\psi} e^{-i\theta_i}\big).

Each oscillator only feels the mean field, which in turn is made of the oscillators: r ≈ 0 means incoherence,
r = 1 perfect synchrony. With a symmetric unimodal frequency density g, the incoherent state loses stability at
K_c = 2 / (π g(0)) (Kuramoto). For the Lorentzian g(ω) = (γ/π)/(ω² + γ²), K_c = 2γ, and Ott and Antonsen (2008)
proved that the N → ∞ dynamics reduces exactly to one equation for r, whose stable state is r = √(1 − K_c/K).

A ring with nonlocal coupling G and a phase lag α (Kuramoto and Battogtokh 2002, Abrams and Strogatz 2004),

    \dot\theta_i = \omega - \frac{2\pi}N \sum_j G(x_i - x_j)\,\sin(\theta_i - \theta_j + \alpha),

can split identical oscillators into a locked domain and a drifting one: a chimera state.
"""

import math

import torch


def order_parameter(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""r e^{iψ} = (1/N) Σ_j e^{iθ_j}   (j summed).   (..., N) → r (...), ψ (...)"""
    z = torch.exp(1j * theta).mean(-1)
    return z.abs(), z.angle()


def velocity(theta: torch.Tensor, omega: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""θ̇_i = ω_i + K Im(z e^{−iθ_i}) with z = r e^{iψ}: one batch member per coupling K.
    (B, N), (N,) or (B, N), (B, 1) → (B, N)"""
    z = torch.exp(1j * theta).mean(-1, keepdim=True)  # (B, 1) complex mean field
    return omega + k * (z * torch.exp(-1j * theta)).imag


def pairwise_velocity(theta: torch.Tensor, omega: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""Slow reference: θ̇_i = ω_i + (K/N) Σ_j sin(θ_j − θ_i)   (j summed over an (N, N) matrix).
    (B, N), (N,) or (B, N), (B, 1) → (B, N)"""
    s = torch.sin(theta[:, None, :] - theta[:, :, None])  # (B, N_i, N_j): sin(θ_j − θ_i)
    return omega + k / theta.shape[-1] * torch.einsum("bij->bi", s)


def critical_coupling(distribution: str, width: float) -> float:
    """K_c = 2 / (π g(0)): Lorentzian half-width γ → 2γ; Gaussian standard deviation σ → 2σ√(2π)/π ≈ 1.596 σ."""
    g0 = 1 / (math.pi * width) if distribution == "lorentzian" else 1 / (width * math.sqrt(2 * math.pi))
    return 2 / (math.pi * g0)


def ott_antonsen_r(k: torch.Tensor, gamma: float) -> torch.Tensor:
    """Exact N → ∞ order parameter for Lorentzian frequencies: r = √(1 − 2γ/K) above K_c = 2γ, else 0.   (B,) → (B,)"""
    return (1 - 2 * gamma / k).clamp(min=0).sqrt()


def ring_kernel(x: torch.Tensor, a: float) -> torch.Tensor:
    """G_ij = (1 + A cos(x_i − x_j)) / 2π on the ring x ∈ [−π, π).   (N,) → (N, N)"""
    return (1 + a * torch.cos(x[:, None] - x[None, :])) / (2 * math.pi)


def ring_velocity(theta: torch.Tensor, g: torch.Tensor, alpha: float, omega: float) -> torch.Tensor:
    r"""θ̇_i = ω − Im(e^{i(θ_i + α)} conj(Z_i)), Z_i = (2π/N) Σ_j G_ij e^{iθ_j}   (j summed), because
    sin(θ_i − θ_j + α) = Im(e^{i(θ_i+α)} e^{−iθ_j}).   (N,), (N, N) → (N,)"""
    z = (2 * math.pi / theta.shape[-1]) * torch.einsum("ij,j->i", g.to(torch.complex128), torch.exp(1j * theta))
    return omega - (torch.exp(1j * (theta + alpha)) * z.conj()).imag


def local_order(theta: torch.Tensor, half_width: int) -> torch.Tensor:
    """|mean of e^{iθ_j}| over the 2w + 1 nearest neighbours on the ring: ≈ 1 in a locked domain.   (N,) → (N,)"""
    z = torch.exp(1j * theta)
    window = torch.stack([torch.roll(z, s) for s in range(-half_width, half_width + 1)])  # (2w+1, N)
    return window.mean(0).abs()
