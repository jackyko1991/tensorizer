"""Build the concrete tensors: the grid, the density profile and the incoming pulse."""

import math

import torch

TENSION = 1.0
MU1 = 1.0
X0 = -0.5  # starting position of the pulse


def grid(n_points: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """x of shape (N,) from −1 to 1 (fixed ends) and the spacing Δx."""
    return torch.linspace(-1.0, 1.0, n_points, device=device), 2.0 / (n_points - 1)


def density(x: torch.Tensor, mu1: float, mu2: float) -> torch.Tensor:
    """μ_i of shape (N,): μ1 for x < 0, μ2 for x > 0 and the average on the knot."""
    mu = torch.where(x < 0, mu1, mu2)
    return torch.where(x.abs() < 1e-9, 0.5 * (mu1 + mu2), mu)


def right_moving_pulse(x: torch.Tensor, width: float, c1: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A Gaussian u(x, t) = exp(−(x − x0 − c1 t)² / 2σ²) at t = 0 and t = −Δt: two (N,) tensors.

    Giving the leap-frog both time levels launches the pulse to the right only.
    """

    def gauss(t: float) -> torch.Tensor:
        return torch.exp(-((x - X0 - c1 * t) ** 2) / (2 * width**2))

    return gauss(-dt), gauss(0.0)


def speed(mu: float) -> float:
    """c = √(T / μ)."""
    return math.sqrt(TENSION / mu)
