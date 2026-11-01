"""Build the concrete tensors: radial grid points and the cylindrical Laplacian matrix."""

import torch


def grid(n: int, R: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """N + 1 radial points r_i = i Δr from the axis (i = 0) to the wall (i = N), and Δr = R/N."""
    dr = R / n
    return dr * torch.arange(n + 1, device=device, dtype=torch.float64), dr


def radial_laplacian(r: torch.Tensor, dr: float) -> torch.Tensor:
    """(N, N) matrix for (1/r) ∂_r(r ∂_r u) acting on the unknowns u_0 … u_{N-1}; the wall value u_N = 0.

    Interior i:  [r_{i+½}(u_{i+1} - u_i) - r_{i-½}(u_i - u_{i-1})] / (r_i Δr²)
    Axis i = 0:  the symmetric limit 2 ∂_r² u = 4 (u_1 - u_0) / Δr²
    Row N-1 simply drops its u_N term, because u_N = 0 (no slip).
    """
    n = len(r) - 1
    L = torch.zeros(n, n + 1, dtype=r.dtype, device=r.device)  # one extra column for the wall, removed below
    L[0, 0], L[0, 1] = -4 / dr**2, 4 / dr**2
    for i in range(1, n):
        outer, inner = r[i] + dr / 2, r[i] - dr / 2  # r_{i+½}, r_{i-½}
        L[i, i + 1] = outer / (r[i] * dr**2)
        L[i, i - 1] = inner / (r[i] * dr**2)
        L[i, i] = -(outer + inner) / (r[i] * dr**2)
    return L[:, :n]


def at_rest(n: int, device: torch.device) -> torch.Tensor:
    """The unknowns u_0 … u_{N-1} are all 0 at t = 0: shape (N,)."""
    return torch.zeros(n, device=device, dtype=torch.float64)
