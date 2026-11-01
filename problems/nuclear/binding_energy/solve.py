"""Read off the physics from the (Z, N) table: the most bound isobar for each A and the B/A peak."""

import torch


def most_stable(b: torch.Tensor, z: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For every mass number A, the Z with the largest binding energy among the isobars on the grid.

    Returns (A (K,), Z_best (K,), (B/A)_best (K,)). Isobars are the anti-diagonals Z + N = A of the table;
    cells with a different A are masked with −∞ before the max over Z.
    """
    mass = (z + n).long()  # (Z, N) mass number of each cell
    a_values = torch.arange(2, int(mass.max()) + 1, device=b.device)  # (K,)
    on_isobar = mass[None] == a_values[:, None, None]  # (K, Z, N)
    masked = torch.where(on_isobar, b[None], torch.tensor(-torch.inf, dtype=b.dtype, device=b.device))
    best_b, flat = masked.flatten(1).max(dim=1)  # best over all cells with this A
    z_best = z.flatten()[flat // b.shape[1]]
    keep = torch.isfinite(best_b) & (best_b > 0)
    a_values, z_best, best_b = a_values[keep].double(), z_best[keep], best_b[keep]
    return a_values, z_best, best_b / a_values


def peak(mass: torch.Tensor, b_per_a: torch.Tensor) -> tuple[float, float]:
    """(A, B/A) where the binding energy per nucleon is largest."""
    k = torch.argmax(b_per_a)
    return mass[k].item(), b_per_a[k].item()
