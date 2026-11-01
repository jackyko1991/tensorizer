"""Measure noisy DW signals, fit a tensor per voxel, and trace fibres along the principal eigenvector with RK2."""

import torch

from problems._common.progress import steps

from .equation import fit_tensor, invariants, signal


def measure(
    D: torch.Tensor, g: torch.Tensor, b: float, noise: float, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """b = 0 image S0 (V,) and diffusion-weighted images S (V, G) with Rician noise of std `noise` (S0 = 1).

    Rician noise: the scanner measures the magnitude of a complex signal, |S + σ(n₁ + i n₂)|.
    """
    gen = torch.Generator(device=D.device).manual_seed(seed)
    clean = torch.cat([torch.ones(D.shape[0], 1, dtype=D.dtype, device=D.device), signal(D, g, b)], dim=1)
    n1 = torch.randn(clean.shape, generator=gen, dtype=D.dtype, device=D.device)
    n2 = torch.randn(clean.shape, generator=gen, dtype=D.dtype, device=D.device)
    noisy = torch.sqrt((clean + noise * n1) ** 2 + (noise * n2) ** 2)
    return noisy[:, 0], noisy[:, 1:]


def fit(S0: torch.Tensor, S: torch.Tensor, B: torch.Tensor, b: float):
    """Fitted tensors (V, 3, 3) and their invariants: eigenvalues, principal direction, MD, FA."""
    D_fit = fit_tensor(S, S0, B, b)
    return (D_fit, *invariants(D_fit))


def _interpolate(field: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of a voxel field (n, n, ...) at positions pos (P, 2) = (x, y). → (P, ...)"""
    n = field.shape[0]
    x = pos[:, 0].clamp(0, n - 1)
    y = pos[:, 1].clamp(0, n - 1)
    x0 = x.floor().long().clamp(max=n - 2)
    y0 = y.floor().long().clamp(max=n - 2)
    fx, fy = x - x0, y - y0
    shape = (-1,) + (1,) * (field.dim() - 2)
    fx, fy = fx.view(shape), fy.view(shape)
    return (
        field[y0, x0] * (1 - fx) * (1 - fy)
        + field[y0, x0 + 1] * fx * (1 - fy)
        + field[y0 + 1, x0] * (1 - fx) * fy
        + field[y0 + 1, x0 + 1] * fx * fy
    )


def _direction(D_grid: torch.Tensor, pos: torch.Tensor, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """In-plane principal direction (P, 2) at pos, flipped to agree with `previous`, and the local FA (P,)."""
    D = _interpolate(D_grid, pos)  # (P, 3, 3)
    _, e1, _, fa = invariants(D)
    d = e1[:, :2] / e1[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-12)
    sign = torch.sign(torch.einsum("pi,pi->p", d, previous))  # eigenvectors have no sign: keep going forward
    return d * torch.where(sign == 0, 1.0, sign)[:, None], fa


def track(
    D_grid: torch.Tensor, start: torch.Tensor, h: float = 0.3, n_steps: int = 90, fa_stop: float = 0.2
) -> torch.Tensor:
    """Streamlines (S+1, 2P, 2) from every seed in both directions; NaN after a line stops.

    Midpoint rule (RK2):  k₁ = e(x),  k₂ = e(x + h/2 k₁),  x ← x + h k₂.  A line stops where FA < fa_stop
    (CSF, grey matter, or a crossing) or when it leaves the slice.
    """
    n = D_grid.shape[0]
    first, _ = _direction(D_grid, start, torch.ones_like(start))
    pos = torch.cat([start, start])  # (2P, 2)
    heading = torch.cat([first, -first])  # forward and backward
    alive = torch.ones(len(pos), dtype=torch.bool, device=pos.device)
    path = [pos.clone()]
    for _ in steps(n_steps, "tracking steps"):
        k1, fa = _direction(D_grid, pos, heading)
        k2, _ = _direction(D_grid, pos + 0.5 * h * k1, k1)
        new = pos + h * k2
        inside = (new >= 0).all(-1) & (new <= n - 1).all(-1)
        alive = alive & inside & (fa >= fa_stop)
        pos = torch.where(alive[:, None], new, pos)
        heading = torch.where(alive[:, None], k2, heading)
        path.append(torch.where(alive[:, None], pos, torch.full_like(pos, float("nan"))))
    return torch.stack(path)
