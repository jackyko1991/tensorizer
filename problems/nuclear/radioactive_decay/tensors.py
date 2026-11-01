"""Build the concrete tensors: a square grid of nuclei and the time axis."""

import torch


def nuclei(n_side: int, device: torch.device) -> torch.Tensor:
    """s_n = True for every nucleus at t = 0, arranged as an (S, S) grid so we can draw it: shape (S, S) bool."""
    return torch.ones(n_side, n_side, dtype=torch.bool, device=device)


def time_axis(t_max: float, dt: float, device: torch.device) -> torch.Tensor:
    """t_k = k Δt for k = 0 … n_steps: shape (T,)."""
    n_steps = int(round(t_max / dt))
    return torch.arange(n_steps + 1, device=device, dtype=torch.float64) * dt
