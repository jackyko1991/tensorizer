"""Build the concrete tensors: one starting temperature and one cooling constant per cup."""

import torch


def cooling_constants(n: int, k_min: float, k_max: float, device: torch.device) -> torch.Tensor:
    """k_b in 1/min, evenly spread from slow to fast cooling.  Shape (B,)."""
    if n == 1:
        return torch.tensor([k_min], device=device)
    return torch.linspace(k_min, k_max, n, device=device)


def start_temperatures(n: int, device: torch.device) -> torch.Tensor:
    """T_{0,b} in °C: the cups are poured between 95 °C and 70 °C.  Shape (B,)."""
    if n == 1:
        return torch.tensor([90.0], device=device)
    return torch.linspace(95.0, 70.0, n, device=device)
