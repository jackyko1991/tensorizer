"""Time-march KdV in Fourier space and locate the solitons afterwards."""

import torch

from problems._common.progress import steps

from .equation import if_rk4_step


def solve(
    u0: torch.Tensor, k: torch.Tensor, dealias: torch.Tensor, t_max: float, n_frames: int = 241
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (F,), u (F, J)) saved at n_frames evenly spaced times."""
    dt = time_step(u0, k)
    n_steps = int(round(t_max / dt))
    save_every = max(1, n_steps // (n_frames - 1))
    u_hat = torch.fft.fft(u0)
    frames, times = [u0], [0.0]
    for step in steps(n_steps, "IF-RK4 steps"):
        u_hat = if_rk4_step(u_hat, k, dealias, dt)
        if (step + 1) % save_every == 0:
            frames.append(torch.fft.ifft(u_hat).real)
            times.append((step + 1) * dt)
    return torch.tensor(times, dtype=u0.dtype, device=u0.device), torch.stack(frames)


def time_step(u0: torch.Tensor, k: torch.Tensor, safety: float = 2.0) -> float:
    """Δt from the nonlinear term only (6 u k Δt ≲ 2.8 for RK4); the dispersion is handled exactly."""
    return min(0.005, safety / (6 * u0.abs().max().item() * k.abs().max().item()))


def peak_position(x: torch.Tensor, u: torch.Tensor) -> float:
    """Sub-grid position of the highest point: fit a parabola through the top three samples."""
    j = int(torch.argmax(u))
    n, dx = len(u), (x[1] - x[0]).item()
    left, mid, right = u[(j - 1) % n].item(), u[j].item(), u[(j + 1) % n].item()
    offset = 0.5 * (left - right) / (left - 2 * mid + right)
    return x[j].item() + offset * dx


def second_peak_position(x: torch.Tensor, u: torch.Tensor, first: float, exclude: float = 6.0) -> float:
    """Position of the highest point farther than `exclude` from `first` (periodic distance)."""
    box = len(u) * (x[1] - x[0]).item()
    dist = torch.remainder(x - first + box / 2, box) - box / 2
    return peak_position(x, torch.where(dist.abs() > exclude, u, torch.zeros_like(u)))
