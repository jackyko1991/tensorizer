"""No time stepping: the radius of every front is known at every time, so we just tabulate it."""

import torch

from .equation import arrival_times, front_radii


def solve(t_e: torch.Tensor, s: torch.Tensor, obs: torch.Tensor, c: float, t_max: float, n_times: int = 200) -> dict:
    """Return times (T,), radii (T, E) of every front at every time, and arrival times (O, E) at the listeners."""
    times = torch.linspace(0.0, t_max, n_times, device=t_e.device)
    radii = torch.stack([front_radii(t, t_e, c) for t in times.tolist()])  # (T, E)
    return {"times": times, "radii": radii, "arrivals": arrival_times(obs, s, t_e, c)}
