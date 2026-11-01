"""Add up the kicks into paths and measure how fast the cloud spreads."""

import torch

from .equation import fit_diffusion, mean_squared_displacement


def solve(t: torch.Tensor, kicks: torch.Tensor) -> dict:
    """t: (T,), kicks: (T-1, N, 2).  Returns positions (T, N, 2), MSD (T,) and the fitted D."""
    start = torch.zeros_like(kicks[:1])  # (1, N, 2)  everyone starts at the origin
    x = torch.cumsum(torch.cat([start, kicks]), dim=0)  # (T, N, 2)  x_t = sum of kicks up to t

    msd = mean_squared_displacement(x)  # (T,)
    return {"x": x, "msd": msd, "D_fit": fit_diffusion(t, msd, x.shape[-1])}
