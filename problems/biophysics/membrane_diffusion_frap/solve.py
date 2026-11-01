"""Time-step the diffusion equation with RK4, then measure t½ and fit D to the recovery curve."""

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from .equation import rhs, soumpasis, spot_intensity


def time_step(D: torch.Tensor, h: float) -> float:
    """Explicit schemes need dt ≲ h² / (4 D) or the stencil blows up; RK4 with 0.3 h²/D is safely inside."""
    return 0.3 * h**2 / D.max().item()


def solve(c0, D, weights, h: float, mask, t_end: float, n_frames: int):
    """Return (times (T,), spot intensity F (T, B), snapshot times (n_frames,), snapshots (n_frames, B, N, N)).

    All species b diffuse together in one (B, N, N) tensor. The spot intensity is recorded every step,
    the whole membrane only at n_frames times, spaced as k² so the fast early recovery gets more frames.
    """
    dt = time_step(D, h)
    n = int(round(t_end / dt))
    save_at = {round(n * (k / max(1, n_frames - 1)) ** 2) for k in range(n_frames)}
    c = c0
    spot, snaps = [spot_intensity(c, mask)], [c]  # k = 0 is always saved
    for step in steps(n, "diffusion steps"):
        c = rk4_step(lambda u, t: rhs(u, D, weights, h), c, step * dt, dt)
        spot.append(spot_intensity(c, mask))
        if step + 1 in save_at:
            snaps.append(c)
    times = torch.arange(n + 1, device=c0.device) * dt
    snap_times = times[sorted(save_at)]
    return times, torch.stack(spot), snap_times, torch.stack(snaps)


def half_time(times: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """First time each recovery curve f[:, b] reaches ½, linearly interpolated between steps.   (B,)"""
    i = (f >= 0.5).int().argmax(dim=0).clamp_min(1)  # first index above ½ for every b
    b = torch.arange(f.shape[1], device=f.device)
    f0, f1 = f[i - 1, b], f[i, b]
    return times[i - 1] + (0.5 - f0) / (f1 - f0) * (times[i] - times[i - 1])


def fit_diffusion(times: torch.Tensor, f: torch.Tensor, w: float, t_half: torch.Tensor) -> torch.Tensor:
    """Least-squares fit of the Soumpasis curve: try 2000 values of τ_D and keep the best one per species.

    Only the early recovery (t ≤ 3 t½) is fitted: later the finite patch runs out of fresh dye and the
    curve leaves the infinite-membrane formula. With the window W_tb (1 inside, 0 outside)

        error_{bm} = W_tb (f_tb − model_tm)²      t summed,     then D = w² / (4 τ_D).   (B,)
    """
    taus = torch.logspace(-3, 3, 2000, device=f.device, dtype=f.dtype)  # candidate τ_D, s
    model = soumpasis(times, taus)  # (T, M)
    W = (times[:, None] <= 3 * t_half[None, :]).to(f.dtype)  # (T, B)
    # (f − model)² expanded, so no (T, B, M) tensor is ever built
    error = (
        torch.einsum("tb,tb->b", W, f**2)[:, None]
        - 2 * torch.einsum("tb,tb,tm->bm", W, f, model)
        + torch.einsum("tb,tm->bm", W, model**2)
    )
    tau_best = taus[error.argmin(dim=1)]
    return w**2 / (4.0 * tau_best)
