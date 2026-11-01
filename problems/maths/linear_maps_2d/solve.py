"""Build the animation path through all matrices and work out det, singular values and eigenvectors along it."""

import torch

from .equation import interpolate


def path(a: torch.Tensor, frames: int, hold: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each matrix: `frames` steps from I to A, then `hold` frames resting on A.

    a (S, 2, 2) → M (T, 2, 2), the matrix index of each frame (T,), and t of each frame (T,), T = S·(frames + hold).
    """
    move = torch.linspace(0, 1, frames, dtype=a.dtype, device=a.device)
    t = torch.cat([move, torch.ones(hold, dtype=a.dtype, device=a.device)])
    m = torch.cat([interpolate(a_s, t) for a_s in a])
    which = torch.arange(len(a), device=a.device).repeat_interleave(len(t))
    return m, which, t.repeat(len(a))


def singular(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """M = U Σ Vᵀ: the unit circle goes to an ellipse with semi-axes σ₁ u₁ and σ₂ u₂.

    (T, 2, 2) → U (T, 2, 2), σ (T, 2)
    """
    u, sigma, _ = torch.linalg.svd(m)
    return u, sigma


def real_eigenvectors(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues λ (k,) and unit eigenvectors in the columns (2, k) of one 2×2 matrix; k = 0 if they are complex."""
    lam, vec = torch.linalg.eig(a)
    if (lam.imag.abs() > 1e-9).any():  # complex pair: every direction turns (a rotation)
        return lam.real[:0], vec.real[:, :0]
    return lam.real, vec.real


def circle_axes(y_circle: torch.Tensor) -> torch.Tensor:
    """Longest and shortest distance from the origin to the image of the unit circle: (…, K, 2) → (…, 2)."""
    r = y_circle.norm(dim=-1)
    return torch.stack([r.amax(dim=-1), r.amin(dim=-1)], dim=-1)
