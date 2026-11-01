"""Step all N bodies forward together with velocity Verlet (kick – drift – kick)."""

import torch

from .equation import acceleration


def solve(
    m: torch.Tensor, x0: torch.Tensor, v0: torch.Tensor, G: float, eps: float, t_max: float, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return times (T,), positions (T, N, 3) and velocities (T, N, 3).

    Velocity Verlet is symplectic: the energy error oscillates but does not drift,
    which is why it is the standard choice for orbits. One force evaluation per step.
    """
    n_steps = int(round(t_max / dt))
    x, v = x0.clone(), v0.clone()
    a = acceleration(x, m, G, eps)  # (N, 3)
    xs, vs = [x], [v]
    for _ in range(n_steps):
        v_half = v + 0.5 * dt * a  # kick
        x = x + dt * v_half  # drift
        a = acceleration(x, m, G, eps)
        v = v_half + 0.5 * dt * a  # kick
        xs.append(x)
        vs.append(v)
    times = torch.arange(n_steps + 1, device=x0.device, dtype=x0.dtype) * dt
    return times, torch.stack(xs), torch.stack(vs)
