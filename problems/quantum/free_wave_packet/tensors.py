"""Build the concrete tensors: periodic grid, its wavenumbers and the initial Gaussian wave packet."""

import math

import torch


def grid(n: int, length: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Periodic grid x ∈ [−length/2, length/2) and its FFT wavenumbers k. Returns (x (N,), k (N,), dx)."""
    dx = length / n
    x = -length / 2 + dx * torch.arange(n, dtype=torch.float64, device=device)  # (N,)
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64, device=device)  # (N,) in FFT order
    return x, k, dx


def gaussian_packet(x: torch.Tensor, x0: float, sigma0: float, k0: float, dx: float) -> torch.Tensor:
    """ψ_x = exp(−(x−x0)²/(4σ0²) + i k0 x), normalised to Σ|ψ|² dx = 1. σ0 is the width of |ψ|². Shape (N,) complex."""
    psi = torch.exp(-((x - x0) ** 2) / (4 * sigma0**2) + 1j * k0 * x)
    return psi / torch.sqrt((psi.abs() ** 2).sum() * dx)
