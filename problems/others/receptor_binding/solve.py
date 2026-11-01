"""Integrate binding with RK4: an association phase (ligand flowing) followed by a dissociation phase (buffer wash)."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def time_step(L: torch.Tensor, k_on: torch.Tensor, k_off: torch.Tensor) -> float:
    """Keep dt well below 1 / k_obs of the fastest (b, c) pair, so RK4 stays accurate."""
    k_fastest = (k_on.max() * L.max() + k_off.max()).item()
    return min(0.05, 0.5 / k_fastest)


def solve(
    y0: torch.Tensor,
    L: torch.Tensor,
    k_on: torch.Tensor,
    k_off: torch.Tensor,
    t_assoc: float,
    t_dissoc: float,
    depletion: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, 2, B, C)) with trajectory[:, 0] = R and trajectory[:, 1] = C.

    Every receptor variant b and every concentration c advances together in one (2, B, C) tensor.
    """
    dt = time_step(L, k_on, k_off)
    n_assoc = int(round(t_assoc / dt))
    n_dissoc = int(round(t_dissoc / dt))

    # 1) association: ligand at concentration L_c
    on = integrate(lambda y, t: rhs(y, L, k_on, k_off, depletion), y0, dt, n_assoc)
    # 2) dissociation: the buffer wash removes all free ligand, so complexes can only fall apart
    buffer = torch.zeros_like(L)
    off = integrate(lambda y, t: rhs(y, buffer, k_on, k_off, False), on[-1], dt, n_dissoc)

    traj = torch.cat([on, off[1:]])  # (T, 2, B, C)
    times = torch.arange(traj.shape[0], device=y0.device) * dt
    return times, traj
