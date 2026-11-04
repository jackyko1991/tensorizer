"""Build the concrete tensors: the thin triangles of the Perron tree, δ-separated directions and tube centres."""

import math

import torch

HALF_WIDTH = 1 / math.sqrt(3)  # base [−h, h] at distance 1 from the apex: a 60° triangle, all sides ≥ 1


def triangle_bases(k: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """Left ends b_i of the N = 2^k bases [b_i, b_i + w₀] and their width w₀.   → (N,), float"""
    n = 2**k
    w0 = 2 * HALF_WIDTH / n
    return -HALF_WIDTH + w0 * torch.arange(n, dtype=torch.float64, device=device), w0


def directions(n: int, delta: float, device: torch.device) -> torch.Tensor:
    """About δ-separated unit vectors covering every line direction in ℝⁿ (one hemisphere).

    A grid of spacing δ on each face x_j = 1 of the cube [−1, 1]ⁿ, pushed out to the sphere: n·(2/δ + 1)^{n−1}
    directions, neighbours between δ/√n and δ apart.   → (D, n)"""
    g = torch.arange(-1, 1 + 1e-9, delta, dtype=torch.float64, device=device)  # (2/δ + 1,)
    face = torch.cartesian_prod(*([g] * (n - 1))).reshape(-1, n - 1)  # (F, n − 1)
    ones = torch.ones(len(face), 1, dtype=torch.float64, device=device)
    dirs = torch.cat([torch.cat([face[:, :j], ones, face[:, j:]], dim=1) for j in range(n)])  # (n·F, n)
    return dirs / dirs.norm(dim=1, keepdim=True)


def centres(arrangement: str, n_tubes: int, n: int, seed: int, device: torch.device) -> torch.Tensor:
    """Tube centres: all at the origin (a "bush": a Kakeya set shaped like a ball) or uniform in [−1, 1]ⁿ."""
    if arrangement == "bush":
        return torch.zeros(n_tubes, n, dtype=torch.float64, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (2 * torch.rand(n_tubes, n, generator=gen, dtype=torch.float64) - 1).to(device)


def rotation(angle: float) -> torch.Tensor:
    """2×2 rotation matrix."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
