"""Time-march NLS with the split-step Fourier method."""

import torch

from problems._common.progress import steps

from .equation import split_step


def solve(
    psi0: torch.Tensor, half_linear: torch.Tensor, dt: float, t_max: float, n_frames: int = 251
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (F,), ψ (F, J) complex) at n_frames evenly spaced times."""
    n_steps = int(round(t_max / dt))
    save_every = max(1, n_steps // (n_frames - 1))
    psi = psi0
    frames, times = [psi0], [0.0]
    for step in steps(n_steps, "split steps"):
        psi = split_step(psi, half_linear, dt)
        if (step + 1) % save_every == 0:
            frames.append(psi)
            times.append((step + 1) * dt)
    return torch.tensor(times, dtype=torch.float64, device=psi0.device), torch.stack(frames)
