"""Build the concrete tensors: the comoving galaxy grid and the time axis.

Units: Mpc for distances, Gyr for times, km/s/Mpc for H_0 in the parameters (converted to 1/Gyr here).
"""

import torch

from .equation import KMS_PER_MPC_GYR


def hubble_constant(H0_kms_mpc: float) -> float:
    """H_0 in 1/Gyr (70 km/s/Mpc ≈ 0.0716 /Gyr, so 1/H_0 ≈ 14 Gyr)."""
    return H0_kms_mpc / KMS_PER_MPC_GYR


def galaxy_grid(n: int, spacing: float, device: torch.device) -> torch.Tensor:
    """Comoving coordinates q_{gi} of an n × n grid centred on the origin: shape (n², 2), in Mpc."""
    k = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * spacing  # (n,)
    qx, qy = torch.meshgrid(k, k, indexing="xy")
    return torch.stack([qx.flatten(), qy.flatten()], dim=-1).to(device)


def nearest_galaxy(q: torch.Tensor, x: float, y: float) -> int:
    """Index of the grid galaxy closest to the comoving point (x, y)."""
    target = torch.tensor([x, y], dtype=q.dtype, device=q.device)
    return int((q - target).norm(dim=-1).argmin())


def time_axis(t_start: float, t_end: float, n: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(t_start, t_end, n, device=device)
