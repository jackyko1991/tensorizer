"""Build the concrete tensors: the time grid and every random kick of every walker."""

import torch

from .equation import kick_size

DIM = 2  # walkers move in the plane


def times(t_max: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """t_t = 0, Δt, 2Δt, ..., t_max.  Shape (S + 1,)."""
    return torch.linspace(0, t_max, n_steps + 1, device=device)


def random_kicks(n_steps: int, n_walkers: int, D: float, dt: float, device: torch.device, seed: int = 0):
    """√(2DΔt) ξ_{sni} with ξ standard normal.  Shape (S, N, 2).  Fixed seed → same picture every run."""
    gen = torch.Generator().manual_seed(seed)
    xi = torch.randn(n_steps, n_walkers, DIM, generator=gen)  # (S, N, 2)
    return (kick_size(D, dt) * xi).to(device)
