"""Solve the eigenvalue problem and put the first few mode shapes back on the square grid."""

import torch

from .equation import laplacian_2d, modes, restrict


def solve(D: torch.Tensor, inside: torch.Tensor, c: float, n_modes: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ω (P,) for all modes and the first n_modes shapes on the grid: (n_modes, n, n), zero outside the drum."""
    n = D.shape[0]
    lap = restrict(laplacian_2d(D), inside)  # (P, P)
    omega, phi = modes(lap, c)
    shapes = torch.zeros(n_modes, n * n, dtype=phi.dtype, device=phi.device)
    shapes[:, inside] = phi[:, :n_modes].T  # scatter each mode back onto the full grid
    shapes = shapes.view(n_modes, n, n)
    # sign convention: make the largest swing of each mode positive (eigh may return −φ)
    peak = shapes.flatten(1).gather(1, shapes.flatten(1).abs().argmax(1, keepdim=True))
    return omega, shapes * torch.sign(peak)[:, :, None]
