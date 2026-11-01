"""Build the concrete tensors: the grid, the anharmonic well, the Hamiltonian matrix and the start packets."""

import math

import torch


def grid(n: int, L: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """N interior points of the box [−L/2, L/2]; the walls are left out (ψ = 0 there). Returns (x (N,), dx)."""
    dx = L / (n + 1)
    x = -L / 2 + dx * torch.arange(1, n + 1, dtype=torch.float64, device=device)  # (N,)
    return x, dx


def potential(x: torch.Tensor, g: float) -> torch.Tensor:
    """V_x = ½ x² + g x⁴: a spring that stiffens with amplitude (g = 0 is the plain harmonic well). Shape (N,)."""
    return 0.5 * x**2 + g * x**4


def second_derivative(n: int, dx: float, device: torch.device) -> torch.Tensor:
    """D2_{xy}: −2 on the diagonal, 1 on the two neighbouring diagonals, all over dx². Shape (N, N)."""
    main = -2.0 * torch.ones(n, dtype=torch.float64, device=device)  # (N,)
    side = torch.ones(n - 1, dtype=torch.float64, device=device)  # (N-1,)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / dx**2


def hamiltonian(x: torch.Tensor, dx: float, V: torch.Tensor) -> torch.Tensor:
    """H_{xy} = −½ D2_{xy} + V_x δ_{xy}. Shape (N, N), real symmetric."""
    return -0.5 * second_derivative(len(x), dx, x.device) + torch.diag(V)


def packets(x: torch.Tensor, x0s: list[float]) -> torch.Tensor:
    """ψ_{xb}(0): the harmonic ground-state Gaussian π^{-¼} e^{−(x − x0_b)²/2}, one column per start point.

    Shape (N, B) complex. Each packet starts at rest.
    """
    centres = torch.tensor(x0s, dtype=torch.float64, device=x.device)  # (B,)
    psi = math.pi**-0.25 * torch.exp(-((x[:, None] - centres[None, :]) ** 2) / 2)  # (N, B)
    return psi.to(torch.complex128)
