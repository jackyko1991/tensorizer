"""Build the concrete tensors: decay constants, initial populations and the time axis."""

import math

import torch


def decay_constants(half_lives: list[float], device: torch.device) -> torch.Tensor:
    """λ_i = ln 2 / T_i for the unstable members, followed by λ = 0 for the stable end: shape (M,)."""
    lam = [math.log(2.0) / t for t in half_lives] + [0.0]
    return torch.tensor(lam, dtype=torch.float64, device=device)


def initial_populations(n0: float, n_members: int, device: torch.device) -> torch.Tensor:
    """N_i(0): a pure sample of the parent, nothing else yet: shape (M,)."""
    n = torch.zeros(n_members, dtype=torch.float64, device=device)
    n[0] = n0
    return n


def time_axis(t_max: float, n_times: int, device: torch.device) -> torch.Tensor:
    """t_k evenly spaced from 0 to t_max: shape (T,)."""
    return torch.linspace(0.0, t_max, n_times, dtype=torch.float64, device=device)
