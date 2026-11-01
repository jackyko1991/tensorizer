"""Build the concrete tensors: the grid, the spring potential, the Hamiltonian matrix and the start state."""

import math

import torch


def grid(n: int, L: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """N interior points of the box [−L/2, L/2]; the walls are left out (ψ = 0 there). Returns (x (N,), dx)."""
    dx = L / (n + 1)
    x = -L / 2 + dx * torch.arange(1, n + 1, dtype=torch.float64, device=device)  # (N,)
    return x, dx


def potential(x: torch.Tensor, omega: float) -> torch.Tensor:
    """V_x = ½ ω² x². Shape (N,)."""
    return 0.5 * omega**2 * x**2


def second_derivative(n: int, dx: float, device: torch.device) -> torch.Tensor:
    """D2_{xy}: −2 on the diagonal, 1 on the two neighbouring diagonals, all over dx². Shape (N, N)."""
    main = -2.0 * torch.ones(n, dtype=torch.float64, device=device)  # (N,)
    side = torch.ones(n - 1, dtype=torch.float64, device=device)  # (N-1,)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / dx**2


def hamiltonian(x: torch.Tensor, dx: float, omega: float) -> torch.Tensor:
    """H_{xy} = −½ D2_{xy} + V_x δ_{xy}. Shape (N, N), symmetric."""
    kinetic = -0.5 * second_derivative(len(x), dx, x.device)  # (N, N)
    return kinetic + torch.diag(potential(x, omega))  # V only on the diagonal


def coherent_state(x: torch.Tensor, x0: float, omega: float) -> torch.Tensor:
    """ψ_x(0) = (ω/π)^¼ exp(−ω(x − x0)²/2): the ground state shifted to x0, at rest. Shape (N,) complex."""
    psi = (omega / math.pi) ** 0.25 * torch.exp(-omega * (x - x0) ** 2 / 2)  # (N,) real
    return psi.to(torch.complex128)
