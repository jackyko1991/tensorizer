"""Build the concrete tensors: the time grid and a batch of deliberately wrong trial paths.

float64 so the action can be pushed down to its minimum without rounding noise.
"""

import torch

DTYPE = torch.float64


def time_grid(T: float, n_steps: int, device=None) -> torch.Tensor:
    """t_n = n Δt for n = 0 … N: (N+1,)."""
    return torch.linspace(0.0, T, n_steps + 1, dtype=DTYPE, device=device)


def trial_paths(t: torch.Tensor, q_start: float, q_end: float, n_paths: int, scale: float) -> torch.Tensor:
    """B wrong guesses with the right end points: (B, N+1).

    Path 0 is the straight line between the ends; path b > 0 adds a wiggle a_b sin(k_b π t/T) that vanishes at
    both ends, with k_b = 1, 2, 3, … and alternating sign, so every guess starts in a different place.
    """
    T = t[-1]
    line = q_start + (q_end - q_start) * t / T  # (N+1,)
    b = torch.arange(n_paths, dtype=t.dtype, device=t.device)  # (B,)
    amp = scale * torch.where(b % 2 == 1, 1.0, -1.0) * (b > 0)  # (B,) path 0 has no wiggle
    wiggle = torch.sin(torch.outer(b, torch.pi * t / T))  # (B, N+1)  k_b = b
    return line + amp[:, None] * wiggle
