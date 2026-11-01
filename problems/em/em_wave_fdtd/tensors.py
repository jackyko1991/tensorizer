"""Build the concrete tensors: Levi-Civita symbol, permittivity map, absorbing border and empty fields."""

import itertools

import torch

EVEN = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]  # cyclic permutations of (x, y, z)


def levi_civita(device: torch.device) -> torch.Tensor:
    """ε_ijk of shape (3, 3, 3): +1 for even permutations, −1 for odd ones, 0 when an index repeats."""
    eps = torch.zeros(3, 3, 3, device=device)
    for i, j, k in itertools.permutations(range(3)):
        eps[i, j, k] = 1.0 if (i, j, k) in EVEN else -1.0
    return eps


def permittivity(nx: int, ny: int, eps_block: float, device: torch.device) -> torch.Tensor:
    """ε_r(x, y) of shape (Nx, Ny): vacuum (1) with a dielectric block in the right half."""
    eps_r = torch.ones(nx, ny, device=device)
    eps_r[int(0.55 * nx) : int(0.8 * nx), int(0.25 * ny) : int(0.85 * ny)] = eps_block
    return eps_r


def damping(nx: int, ny: int, width: int, strength: float, device: torch.device) -> torch.Tensor:
    """Absorbing border, shape (Nx, Ny): 1 inside, falling smoothly to 1 − strength at the walls.

    Multiplying the fields by it every step soaks up outgoing waves. The profile rises quadratically, so the
    wave meets no sharp edge to reflect from. Set ny = 1 for a 1D run: then only the x walls absorb.
    """

    def profile(n: int) -> torch.Tensor:
        depth = torch.arange(n, device=device, dtype=torch.float32)
        distance_in = torch.minimum(depth, n - 1 - depth)  # cells from the nearest wall
        inside = ((width - distance_in) / width).clamp(min=0.0)  # 0 inside, → 1 at the wall
        return 1.0 - strength * inside**2

    if ny == 1:
        return profile(nx)[:, None]
    return profile(nx)[:, None] * profile(ny)[None, :]


def empty_fields(nx: int, ny: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """E_i and H_i, each of shape (3, Nx, Ny), all zero. TM mode: only E_z, H_x and H_y will become non-zero."""
    return torch.zeros(3, nx, ny, device=device), torch.zeros(3, nx, ny, device=device)
