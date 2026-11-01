"""March the strings forward with leap-frog; the time step obeys the Courant condition c Δt ≤ Δx."""

import torch

from problems._common.progress import steps

from .equation import acceleration, leapfrog_step


def solve(
    u0: torch.Tensor, D: torch.Tensor, c: float, dx: float, t_max: float, courant: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (times (T,), trajectory (T, B, N) with the fixed ends, dt).

    u0: (B, N) starting shapes, released from rest.
    """
    dt = courant * dx / c
    n_steps = int(round(t_max / dt))
    u = u0[:, 1:-1]  # (B, M) interior points only
    # Released from rest: choose u^{-1} so that the first step is u^1 = u^0 + ½ Δt² ü^0
    u_prev = u + 0.5 * dt**2 * acceleration(u, D, c)

    frames = [u]
    for _ in steps(n_steps, "time steps"):
        u, u_prev = leapfrog_step(u, u_prev, D, c, dt), u
        frames.append(u)

    interior = torch.stack(frames)  # (T, B, M)
    traj = torch.nn.functional.pad(interior, (1, 1))  # put the fixed ends u = 0 back: (T, B, N)
    times = torch.arange(n_steps + 1, device=u0.device) * dt
    return times, traj, dt
