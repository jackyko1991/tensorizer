"""Evaluate the relaxation curves and the spin-echo signal tensor, and plan a sweep through (TR, TE)."""

import torch

from .equation import image, longitudinal, spin_echo_signal, transverse


def relaxation_curves(
    T1: torch.Tensor, T2: torch.Tensor, PD: torch.Tensor, t_long: torch.Tensor, t_short: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """M_z on the long time axis (S, T) and M_xy on the short one (S, T') with M_0 = PD."""
    return longitudinal(t_long, T1, PD), transverse(t_short, T2, PD)


def nearest(grid: torch.Tensor, value: float) -> int:
    """Index of the grid point closest to value."""
    return int(torch.argmin((grid - value).abs()))


def sweep(TR: torch.Tensor, TE: torch.Tensor, corners: list[tuple[float, float]], n_frames: int) -> torch.Tensor:
    """Grid indices (F, 2) of a path that visits the (TR, TE) corners in order, moving in straight lines.

    Each corner is held for a few frames so the reader can see that weighting.
    """
    per_leg = max(2, n_frames // (2 * len(corners)))
    hold = per_leg // 2
    path = []
    for (tr0, te0), (tr1, te1) in zip(corners, corners[1:] + corners[:1]):
        path += [(tr0, te0)] * hold
        for k in range(per_leg):
            f = k / per_leg
            path.append((tr0 + f * (tr1 - tr0), te0 + f * (te1 - te0)))
    return torch.tensor([(nearest(TR, a), nearest(TE, b)) for a, b in path])


def solve(
    T1: torch.Tensor,
    T2: torch.Tensor,
    PD: torch.Tensor,
    TR: torch.Tensor,
    TE: torch.Tensor,
    labels: torch.Tensor,
    path: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signal tensor S (S, A, B) and the phantom images along the path (F, N, N)."""
    S = spin_echo_signal(PD, T1, T2, TR, TE)  # (S, A, B)
    S_path = S[:, path[:, 0], path[:, 1]]  # (S, F)   pick the (TR, TE) of every frame
    frames = image(labels, S_path).permute(2, 0, 1)  # (N, N, F) → (F, N, N)
    return S, frames
