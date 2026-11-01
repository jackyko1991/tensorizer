"""Build the concrete tensors: radial grid, second-derivative matrix, the l values and a 2D slice grid."""

import torch


def radial_grid(n: int, r_max: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """N interior points r_i = i·dr, i = 1..N, dr = r_max/(N+1); u = 0 at r = 0 and r = r_max. Returns (r (N,), dr)."""
    dr = r_max / (n + 1)
    r = dr * torch.arange(1, n + 1, dtype=torch.float64, device=device)  # (N,)
    return r, dr


def second_derivative(n: int, dr: float, device: torch.device) -> torch.Tensor:
    """D2_{ij}: −2 on the diagonal, 1 on the two neighbouring diagonals, all over dr². Shape (N, N)."""
    main = -2.0 * torch.ones(n, dtype=torch.float64, device=device)  # (N,)
    side = torch.ones(n - 1, dtype=torch.float64, device=device)  # (N-1,)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / dr**2


def angular_momenta(n_max: int, device: torch.device) -> torch.Tensor:
    """l = 0, 1, …, n_max − 1: every l that appears in shells up to n_max. Shape (L,)."""
    return torch.arange(n_max, dtype=torch.float64, device=device)


def slice_grid(half_width: float, n_pix: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Points of the xz-plane in [−a, a]²: returns (r (P, P), cos θ (P, P)) with θ measured from +z."""
    s = torch.linspace(-half_width, half_width, n_pix, dtype=torch.float64, device=device)  # (P,)
    Z, X = torch.meshgrid(s, s, indexing="ij")  # rows: z, columns: x
    r = torch.sqrt(X**2 + Z**2).clamp_min(1e-9)
    return r, Z / r
