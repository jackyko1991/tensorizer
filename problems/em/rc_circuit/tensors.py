"""Build the concrete tensors: one resistance per circuit and the starting voltages."""

import torch


def resistances(R_min: float, R_max: float, n: int, device: torch.device) -> torch.Tensor:
    """R_b of shape (B,): n resistances (kΩ) evenly spread from R_min to R_max."""
    if n == 1:
        return torch.tensor([R_min], device=device)
    return torch.linspace(R_min, R_max, n, device=device)


def initial_voltage(n: int, device: torch.device) -> torch.Tensor:
    """V_b(0) of shape (B,): every capacitor starts empty."""
    return torch.zeros(n, device=device)
