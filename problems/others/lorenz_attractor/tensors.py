"""Build the concrete tensors: the linear part A_ij (3, 3), the quadratic part Q_ijk (3, 3, 3)
and a batch of starting points x_bi (B, 3) inside a tiny ball.

float64, because the trajectories start only ~1e-6 apart.
"""

import math

import torch

DTYPE = torch.float64


def lorenz_tensors(sigma: float, rho: float, beta: float, device: torch.device | None = None):
    r"""(A (3, 3), Q (3, 3, 3)) with \dot x_i = A_{ij} x_j + Q_{ijk} x_j x_k."""
    A = torch.tensor(
        [
            [-sigma, sigma, 0.0],  # ẋ = σ(y − x)
            [rho, -1.0, 0.0],  # ẏ = ρx − y  (− xz is quadratic)
            [0.0, 0.0, -beta],  # ż = − βz   (+ xy is quadratic)
        ],
        dtype=DTYPE,
        device=device,
    )
    Q = torch.zeros(3, 3, 3, dtype=DTYPE, device=device)
    Q[1, 0, 2] = -1.0  # ẏ gets − x z
    Q[2, 0, 1] = 1.0  # ż gets + x y
    return A, Q


def fixed_points(sigma: float, rho: float, beta: float, device: torch.device | None = None) -> torch.Tensor:
    """The origin and, for ρ > 1, the two wing centres C± = (±c, ±c, ρ − 1) with c = √(β(ρ − 1)).  → (P, 3)"""
    points = [[0.0, 0.0, 0.0]]
    if rho > 1:
        c = math.sqrt(beta * (rho - 1))
        points += [[c, c, rho - 1], [-c, -c, rho - 1]]
    return torch.tensor(points, dtype=DTYPE, device=device)


def initial_ball(center: torch.Tensor, radius: float, n: int, seed: int = 0) -> torch.Tensor:
    """n starting points at distance `radius` from `center`, in random directions.  center: (3,) → (n, 3)"""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, 3, generator=g, dtype=DTYPE)
    u = u / u.norm(dim=1, keepdim=True)  # unit vectors
    return center + radius * u.to(center.device)
