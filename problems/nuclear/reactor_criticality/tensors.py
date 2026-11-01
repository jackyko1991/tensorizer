"""Build the concrete tensors: the slab grid, the discrete Laplacian and the starting flux guess."""

import torch


def grid(width: float, n: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """Interior points x_i = h … L − h (the flux is 0 at both edges): (N,), spacing h."""
    h = width / (n + 1)
    return torch.arange(1, n + 1, dtype=torch.float64, device=device) * h, h


def laplacian(n: int, h: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    """Δ_ij φ_j ≈ φ''(x_i): tridiagonal (1, −2, 1)/h². With a tensor of S spacings h → (S, N, N), else (N, N)."""
    main = torch.full((n,), -2.0, dtype=torch.float64, device=device)
    off = torch.ones(n - 1, dtype=torch.float64, device=device)
    stencil = torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)  # (N, N)
    if isinstance(h, torch.Tensor):
        return stencil / h[:, None, None] ** 2
    return stencil / h**2


def initial_guess(x: torch.Tensor, width: float) -> torch.Tensor:
    """A deliberately lopsided start, φ⁽⁰⁾ = 0.2 + x/L, so we can watch it relax to the fundamental mode: (N,)."""
    return 0.2 + x / width
