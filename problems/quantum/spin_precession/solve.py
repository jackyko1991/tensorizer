"""Evolve the spinor step by step with the exact one-step propagators, recording the Bloch vector."""

import torch

from . import tensors
from .equation import bloch_vector, hamiltonian, propagator, step


def solve(
    psi0: torch.Tensor,
    sigma: torch.Tensor,
    gamma: float,
    B0: float,
    B1: float,
    drive_freq: float,
    t_max: float,
    dt: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (times (T,), ψ (T, 2) complex, s (T, 3)).

    All T−1 propagators are built in one batch: the field at each step's midpoint → H (T−1, 2, 2) → U.
    Only the product U_{n} ⋯ U_1 ψ0 has to run as a loop, because each step needs the previous state.
    """
    n_steps = int(round(t_max / dt))
    times = dt * torch.arange(n_steps + 1, dtype=torch.float64, device=psi0.device)  # (T,)
    B = tensors.field(times[:-1] + dt / 2, B0, B1, drive_freq)  # (T-1, 3) midpoint rule
    U = propagator(hamiltonian(B, sigma, gamma), dt)  # (T-1, 2, 2)

    states = [psi0]
    for n in range(n_steps):
        states.append(step(U[n], states[-1]))
    psi = torch.stack(states)  # (T, 2)
    s = torch.stack([bloch_vector(p, sigma) for p in psi])  # (T, 3)
    return times, psi, s
