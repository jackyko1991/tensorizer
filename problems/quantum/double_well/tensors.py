"""Build the concrete tensors: the grid, the double-well potential, the Hamiltonian and the start coefficients."""

import math

import torch


def grid(n: int, L: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """N interior points of the box [−L/2, L/2], symmetric about x = 0 (ψ = 0 at the walls). Returns (x (N,), dx)."""
    dx = L / (n + 1)
    x = -L / 2 + dx * torch.arange(1, n + 1, dtype=torch.float64, device=device)  # (N,)
    return x, dx


def double_well(x: torch.Tensor, V0: float, a: float) -> torch.Tensor:
    """V_x = V0 (x²/a² − 1)²: zero at the minima x = ±a, height V0 at x = 0. Shape (N,)."""
    return V0 * (x**2 / a**2 - 1) ** 2


def second_derivative(n: int, dx: float, device: torch.device) -> torch.Tensor:
    """D2_{xy}: −2 on the diagonal, 1 on the two neighbouring diagonals, all over dx². Shape (N, N)."""
    main = -2.0 * torch.ones(n, dtype=torch.float64, device=device)  # (N,)
    side = torch.ones(n - 1, dtype=torch.float64, device=device)  # (N-1,)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / dx**2


def hamiltonian(x: torch.Tensor, dx: float, V: torch.Tensor) -> torch.Tensor:
    """H_{xy} = −½ D2_{xy} + V_x δ_{xy}. Shape (N, N), symmetric."""
    return -0.5 * second_derivative(len(x), dx, x.device) + torch.diag(V)


def left_state(n: int, device: torch.device) -> torch.Tensor:
    """c_n = (1, 1, 0, …, 0)/√2: equal mix of the even and odd ground doublet. Shape (N,) complex."""
    c = torch.zeros(n, dtype=torch.complex128, device=device)
    c[:2] = 1 / math.sqrt(2)
    return c
