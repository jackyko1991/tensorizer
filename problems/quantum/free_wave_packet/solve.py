"""Evaluate the exact free evolution at every saved time, and the textbook centre and width."""

import torch

from .equation import evolve


def solve(psi0: torch.Tensor, k: torch.Tensor, t_max: float, n_steps: int = 240) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), ψ (T, N) complex). No time stepping: the k-space solution is exact at any t."""
    times = torch.linspace(0.0, t_max, n_steps + 1, dtype=torch.float64, device=psi0.device)  # (T,)
    return times, evolve(psi0, times, k)


def exact_centre(times: torch.Tensor, x0: float, k0: float) -> torch.Tensor:
    """⟨x⟩(t) = x0 + k0 t: the packet moves at the group velocity v_g = k0 (ħ = m = 1). Shape (T,)."""
    return x0 + k0 * times


def exact_width(times: torch.Tensor, sigma0: float) -> torch.Tensor:
    """σ(t) = σ0 √(1 + (t / (2σ0²))²) for the standard deviation of |ψ|². Shape (T,)."""
    return sigma0 * torch.sqrt(1 + (times / (2 * sigma0**2)) ** 2)
