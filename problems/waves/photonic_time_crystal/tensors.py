"""Build the concrete tensors: periodic x grid, its wavenumbers, the wave packet and the k list for the bands."""

import math

import torch

BOX = 2 * math.pi * 64  # periodic box length → wavenumber spacing 1/64
N_GRID = 1024
X_START = -30.0  # where the pulse starts
WIDTH = 8.0  # Gaussian envelope width σ


def grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """x (X,) on [-BOX/2, BOX/2) and the FFT wavenumbers k (K = X,). float64 for clean Floquet eigenvalues."""
    dx = BOX / N_GRID
    x = (torch.arange(N_GRID, device=device, dtype=torch.float64) - N_GRID // 2) * dx
    k = 2 * math.pi * torch.fft.fftfreq(N_GRID, d=dx).to(device=device, dtype=torch.float64)
    return x, k


def forward_pulse(x: torch.Tensor, k_carrier: float, n: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A right-moving wave packet: D = e^{-(x-x0)²/2σ²} cos(k_c (x - x0)), B = D/n.  Each (X,)."""
    s = x - X_START
    D = torch.exp(-(s**2) / (2 * WIDTH**2)) * torch.cos(k_carrier * s)
    return D, D / n


def band_wavenumbers(k_max: float, n: int, device: torch.device) -> torch.Tensor:
    """The k values for the Floquet band structure (K,), a batch: every k is an independent 2×2 problem."""
    return torch.linspace(0.0, k_max, n + 1, device=device, dtype=torch.float64)[1:]
