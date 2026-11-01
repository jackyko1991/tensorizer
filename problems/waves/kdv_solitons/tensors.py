"""Build the concrete tensors: periodic grid, wavenumbers, the 2/3 de-aliasing mask and two solitons."""

import math

import torch

from .equation import soliton

BOX = 100.0
X_TALL, X_SHORT = 15.0, 30.0  # starting positions


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """x (J,) on [0, BOX), wavenumbers k (K = J,), and Δx. float64 keeps the conserved quantities clean."""
    dx = BOX / n
    x = torch.arange(n, device=device, dtype=torch.float64) * dx
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx).to(device=device, dtype=torch.float64)
    return x, k, dx


def dealias_mask(k: torch.Tensor) -> torch.Tensor:
    """1 for the lowest two thirds of the wavenumbers, 0 above (removes aliasing from u²): (K,)."""
    return (k.abs() < (2 / 3) * k.abs().max()).to(k.dtype)


def two_solitons(x: torch.Tensor, c_tall: float, c_short: float) -> torch.Tensor:
    """u0 (J,): the tall fast soliton behind the short slow one."""
    return soliton(x, c_tall, X_TALL) + soliton(x, c_short, X_SHORT)
