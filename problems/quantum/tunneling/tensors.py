"""Build the concrete tensors: periodic grid, its wavenumbers, the barrier and the initial wave packet."""

import math

import torch


def grid(n: int, length: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Periodic grid x ∈ [−length/2, length/2) and its FFT wavenumbers k. Returns (x (N,), k (N,), dx)."""
    dx = length / n
    x = -length / 2 + dx * torch.arange(n, dtype=torch.float64, device=device)  # (N,)
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64, device=device)  # (N,) in FFT order
    return x, k, dx


def barrier(x: torch.Tensor, V0: float, width: float) -> torch.Tensor:
    """V_x = V0 for |x| < a/2, else 0. Shape (N,)."""
    return torch.where(x.abs() < width / 2, V0, 0.0).to(x.dtype)


def grid_width(x: torch.Tensor, width: float, dx: float) -> float:
    """Width the barrier really has on the grid: a whole number of cells, so not exactly a."""
    return int((x.abs() < width / 2).sum()) * dx


def gaussian_packet(x: torch.Tensor, x0: float, sigma: float, k0: float, dx: float) -> torch.Tensor:
    """ψ_x = exp(−(x−x0)²/(4σ²) + i k0 x), normalised to Σ|ψ|² dx = 1. σ is the width of |ψ|². Shape (N,) complex."""
    psi = torch.exp(-((x - x0) ** 2) / (4 * sigma**2) + 1j * k0 * x)
    return psi / torch.sqrt((psi.abs() ** 2).sum() * dx)
