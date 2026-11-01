"""Build the concrete tensors: the grid, the Hamiltonian matrix and the initial wavefunction."""

import math

import torch


def grid(n: int, L: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """N interior points x_i = i·dx, i = 1..N, dx = L/(N+1). The walls x = 0, L are left out (ψ = 0 there)."""
    dx = L / (n + 1)
    x = dx * torch.arange(1, n + 1, dtype=torch.float64, device=device)  # (N,)
    return x, dx


def second_derivative(n: int, dx: float, device: torch.device) -> torch.Tensor:
    """D2_{xy}: −2 on the diagonal, 1 on the two neighbouring diagonals, all over dx². Shape (N, N)."""
    main = -2.0 * torch.ones(n, dtype=torch.float64, device=device)  # (N,)
    side = torch.ones(n - 1, dtype=torch.float64, device=device)  # (N-1,)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / dx**2


def hamiltonian(n: int, L: float, device: torch.device) -> torch.Tensor:
    """H_{xy} = −½ D2_{xy}  (kinetic energy only: V = 0 inside the box). Shape (N, N), symmetric."""
    _, dx = grid(n, L, device)
    return -0.5 * second_derivative(n, dx, device)


def equal_superposition(n_states: int, n: int, device: torch.device) -> torch.Tensor:
    """c_n = 1/√k for the lowest k states, 0 for the rest. Shape (N,) complex."""
    c = torch.zeros(n, dtype=torch.complex128, device=device)
    c[:n_states] = 1.0 / math.sqrt(n_states)
    return c


def gaussian_packet(x: torch.Tensor, L: float, dx: float) -> torch.Tensor:
    """ψ_x(0): a Gaussian at L/4 moving right with momentum k0 = 40/L, normalised so Σ|ψ|²dx = 1. Shape (N,)."""
    x0, sigma, k0 = 0.25 * L, 0.06 * L, 40.0 / L
    psi = torch.exp(-((x - x0) ** 2) / (4 * sigma**2) + 1j * k0 * x)  # (N,) complex128
    return psi / torch.sqrt((psi.abs() ** 2).sum() * dx)
