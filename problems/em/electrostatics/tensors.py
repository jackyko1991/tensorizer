"""Build the concrete tensors: the grid and the charge density ρ for each layout."""

import torch

# Line charges as (x, y, sign) in the unit square; "capacitor" uses two charged plates instead.
LAYOUTS = {
    "dipole": [(0.35, 0.5, +1), (0.65, 0.5, -1)],
    "quadrupole": [(0.35, 0.35, +1), (0.65, 0.65, +1), (0.35, 0.65, -1), (0.65, 0.35, -1)],
    "same_sign": [(0.35, 0.5, +1), (0.65, 0.5, +1)],
}


def grid(n: int) -> tuple[torch.Tensor, float]:
    """Interior grid coordinates x_b = (b + 1) h of shape (N,), spacing h = 1 / (N + 1)."""
    h = 1.0 / (n + 1)
    return torch.arange(1, n + 1, dtype=torch.float64) * h, h


def charge_density(config: str, n: int, charge: float) -> torch.Tensor:
    """ρ_{ab} of shape (N, N) (row a = y, column b = x).

    A line charge λ sits in a single cell, so its density is λ / h² (the cell area is h²).
    """
    _, h = grid(n)
    rho = torch.zeros(n, n, dtype=torch.float64)
    if config == "capacitor":
        lo, hi = round(0.3 * n), round(0.7 * n)  # plates span 0.3 < y < 0.7
        per_cell = charge / (hi - lo)  # total charge λ on each plate
        rho[lo:hi, round(0.35 * n)] = +per_cell / h**2
        rho[lo:hi, round(0.65 * n)] = -per_cell / h**2
        return rho
    for x, y, sign in LAYOUTS[config]:
        a, b = round(y / h) - 1, round(x / h) - 1  # nearest interior cell
        rho[a, b] += sign * charge / h**2
    return rho
