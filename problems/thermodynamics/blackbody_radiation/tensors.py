"""Build the concrete tensors: the batch of temperatures and the wavelength grid (float64)."""

import torch


def temperatures(T_min: float, T_max: float, n: int, device: torch.device) -> torch.Tensor:
    """T_a in kelvin, sweeping from T_min to T_max.  Shape (A,)."""
    return torch.linspace(T_min, T_max, n, dtype=torch.float64, device=device)


def wavelengths(device: torch.device, n: int = 4000) -> torch.Tensor:
    """λ_b in metres, logarithmically spaced from 10 nm to 100 µm.  Shape (L,).

    Log spacing puts many points near the short-wavelength peak and still reaches far into the long tail.
    """
    return torch.logspace(-8, -4, n, dtype=torch.float64, device=device)
