"""Event-free molecular dynamics: drift, bounce off the walls, collide, repeat."""

import torch

from .equation import bounce_walls, collide, colliding_pairs


def solve(
    x0: torch.Tensor,
    v0: torch.Tensor,
    box: float,
    radius: float,
    t_max: float,
    mass: float = 1.0,
    n_snapshots: int = 121,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return (times (F,), positions (F, N, 2), velocities (F, N, 2), pressure P).

    The time step lets the fastest disk move at most half its radius per step,
    so no collision is missed. P = (momentum given to the walls) / (time × wall length).
    """
    v_max = 3 * v0.norm(dim=1).max().item()  # speeds rarely exceed 3× the initial one
    n_steps = max(1, int(t_max * v_max / (0.5 * radius)))
    dt = t_max / n_steps
    keep = set(torch.linspace(0, n_steps, n_snapshots).round().int().tolist())  # steps to record

    x, v = x0, v0
    times, xs, vs = [0.0], [x0], [v0]
    momentum_to_walls = 0.0
    for step in range(1, n_steps + 1):
        x = x + v * dt  # free flight
        v, impulse = bounce_walls(x, v, box, radius, mass)
        momentum_to_walls += impulse.item()
        a, b = colliding_pairs(x, v, radius)
        if len(a):
            v = collide(x, v, a, b)
        if step in keep:
            times.append(step * dt)
            xs.append(x)
            vs.append(v)

    # The disk centres only reach to within one radius of the walls: effective side L − 2r.
    wall_length = 4 * (box - 2 * radius)
    pressure = momentum_to_walls / (t_max * wall_length)
    return torch.tensor(times), torch.stack(xs), torch.stack(vs), pressure
