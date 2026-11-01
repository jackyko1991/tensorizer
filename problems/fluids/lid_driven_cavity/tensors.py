"""Build the concrete tensors: the square grid, the eigen-decomposed 1D Laplacian and the initial fields."""

import torch


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """N points per side on [0, 1] (walls included) and the spacing h = 1/(N-1)."""
    return torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float64), 1.0 / (n - 1)


def laplacian_eigen(n_interior: int, h: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """1D Dirichlet Laplacian T = tridiag(1, -2, 1)/h² on the interior points, decomposed as T = V diag(Λ) V^T.

    Returns V (n, n) and Λ (n,). Done once; every Poisson solve then costs four small matrix products.
    """
    T = (
        torch.diag(torch.full((n_interior - 1,), 1.0), 1)
        + torch.diag(torch.full((n_interior,), -2.0))
        + torch.diag(torch.full((n_interior - 1,), 1.0), -1)
    ).to(device=device, dtype=torch.float64) / h**2
    lam, V = torch.linalg.eigh(T)
    return V, lam


def at_rest(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """ω = ψ = 0 everywhere: two (N, N) tensors indexed [i (x), j (y)]."""
    zeros = torch.zeros(n, n, device=device, dtype=torch.float64)
    return zeros.clone(), zeros.clone()
