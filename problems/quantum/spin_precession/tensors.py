"""Build the concrete tensors: Pauli tensor, Levi-Civita symbol, the field history and the initial spinor."""

import math

import torch


def pauli(device: torch.device) -> torch.Tensor:
    """σ_{kab}: the three Pauli matrices stacked along k. Shape (3, 2, 2) complex."""
    sx = [[0, 1], [1, 0]]
    sy = [[0, -1j], [1j, 0]]
    sz = [[1, 0], [0, -1]]
    return torch.tensor([sx, sy, sz], dtype=torch.complex128, device=device)


def levi_civita(device: torch.device) -> torch.Tensor:
    """ε_{ijk}: +1 for even permutations of (0, 1, 2), −1 for odd ones, 0 otherwise. Shape (3, 3, 3)."""
    eps = torch.zeros(3, 3, 3, dtype=torch.complex128, device=device)
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[i, j, k] = 1.0
        eps[j, i, k] = -1.0
    return eps


def field(times: torch.Tensor, B0: float, B1: float, drive_freq: float) -> torch.Tensor:
    """B_{tk} = (B1 cos ωt, 0, B0) at every time: static field along z plus a linear drive along x. Shape (T, 3)."""
    bx = B1 * torch.cos(drive_freq * times)  # (T,)
    by = torch.zeros_like(times)
    bz = torch.full_like(times, B0)
    return torch.stack([bx, by, bz], dim=-1)


def tilted_spinor(theta_deg: float, device: torch.device) -> torch.Tensor:
    """ψ_a = (cos θ/2, sin θ/2): spin pointing at angle θ from +z in the xz-plane. Shape (2,) complex."""
    half = math.radians(theta_deg) / 2
    return torch.tensor([math.cos(half), math.sin(half)], dtype=torch.complex128, device=device)
