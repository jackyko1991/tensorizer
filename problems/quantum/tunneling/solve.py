"""March the wave packet with split-step Fourier and measure what gets through the barrier."""

import math

import torch

from .equation import kinetic_phase, potential_phase, probability, split_step


def solve(
    psi0: torch.Tensor, V: torch.Tensor, k: torch.Tensor, t_max: float, dt: float = 0.01, n_saved: int = 240
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), ψ (T, N) complex), keeping about n_saved snapshots.

    P and K do not change in time, so they are built once; every step is two FFTs and three multiplies.
    """
    n_steps = int(round(t_max / dt))
    save_every = max(1, n_steps // n_saved)
    P = potential_phase(V, dt)  # (N,)
    K = kinetic_phase(k, dt)  # (N,)

    psi, snapshots, times = psi0, [psi0], [0.0]
    for step in range(1, n_steps + 1):
        psi = split_step(psi, P, K)
        if step % save_every == 0:
            snapshots.append(psi)
            times.append(step * dt)
    return torch.tensor(times), torch.stack(snapshots)


def transmission(psi: torch.Tensor, x: torch.Tensor, width: float, dx: float) -> torch.Tensor:
    """P_T: probability to the right of the barrier, for each snapshot. (..., N) → (...)"""
    right = (x > width / 2).to(psi.dtype)  # (N,) mask
    return probability(psi * right, dx)


def plane_wave_transmission(E: float, V0: float, width: float) -> float:
    """Textbook T for a plane wave of energy E on a barrier V0 of width a (ħ = m = 1)."""
    if V0 == 0:
        return 1.0
    if abs(E - V0) < 1e-9:
        return 1.0 / (1.0 + V0 * width**2 / 2)
    if E < V0:
        kappa = math.sqrt(2 * (V0 - E))
        return 1.0 / (1.0 + V0**2 * math.sinh(kappa * width) ** 2 / (4 * E * (V0 - E)))
    q = math.sqrt(2 * (E - V0))
    return 1.0 / (1.0 + V0**2 * math.sin(q * width) ** 2 / (4 * E * (E - V0)))
