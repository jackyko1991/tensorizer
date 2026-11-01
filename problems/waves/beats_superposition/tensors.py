"""Build the concrete tensors: frequencies, wavenumbers from the dispersion relation, amplitudes and grids."""

import math

import torch

C0 = 1.0  # phase velocity (m/s) at the reference wavenumber
K0 = 2 * math.pi  # reference wavenumber (rad/m): a 1 Hz wave is 1 m long
POWERS = {"none": 1.0, "water": 0.5, "capillary": 1.5}  # ω ∝ k^p


def angular_frequencies(f1: float, f2: float, n_waves: int, device: torch.device) -> torch.Tensor:
    """ω_s = 2π f_s with f_s evenly spread from f1 to f2: (S,)."""
    return 2 * math.pi * torch.linspace(f1, f2, n_waves, device=device)


def wavenumbers(omega: torch.Tensor, dispersion: str) -> torch.Tensor:
    """Invert ω = c0 k0 (k / k0)^p  →  k = k0 (ω / (c0 k0))^{1/p}: (S,)."""
    p = POWERS[dispersion]
    return K0 * (omega / (C0 * K0)) ** (1.0 / p)


def amplitudes(n_waves: int, device: torch.device) -> torch.Tensor:
    """A_s = 1/S so the sum never exceeds 1: (S,)."""
    return torch.full((n_waves,), 1.0 / n_waves, device=device)


def grids(x_max: float, t_max: float, device: torch.device, nx: int = 400, dt: float = 0.02):
    """x (J,) for the snapshot and t (N,) for the time trace."""
    x = torch.linspace(0.0, x_max, nx, device=device)
    t = torch.arange(0.0, t_max + 1e-9, dt, device=device)
    return x, t
