"""Build the concrete tensors: the periodic grid, its wavenumbers and the starting wave packet."""

import math

import torch

G = 9.81  # m/s²
BOX = 400.0  # length of the periodic box (m)
X0 = 60.0  # starting centre of the packet (m)


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """x (J,) on [0, BOX), the matching FFT wavenumbers k (K = J,) in rad/m, and Δx."""
    dx = BOX / n
    x = torch.arange(n, device=device, dtype=torch.float64) * dx
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx).to(device=device, dtype=torch.float64)
    return x, k, dx


def wave_packet(x: torch.Tensor, x0: float, sigma: float, k0: float) -> torch.Tensor:
    """a_j(0) = exp(−(x − x0)² / 2σ²) · e^{i k0 x}: (J,) complex. Its real part is the water surface."""
    envelope = torch.exp(-((x - x0) ** 2) / (2 * sigma**2))
    return envelope * torch.exp(1j * k0 * x)
