"""Build the concrete tensors: the twins' world lines and a grid of reference lines, all as events x^μ = (t, x)."""

import torch


def twin_world_lines(v: float, distance: float, n: int = 801) -> tuple[torch.Tensor, torch.Tensor]:
    """Events (n, 2) along the home twin (x = 0) and the travelling twin, sampled at the same coordinate times.

    The traveller flies at speed v to a star at x = distance, turns around instantly and flies back.
    """
    T = 2 * distance / v  # coordinate time of the reunion
    t = torch.linspace(0.0, T, n, dtype=torch.float64)
    home = torch.stack([t, torch.zeros_like(t)], dim=-1)
    x = v * torch.minimum(t, T - t)  # out, then back
    traveller = torch.stack([t, x], dim=-1)
    return home, traveller


def frame_grid(t_max: float, x_min: float, x_max: float, step: float = 1.0, n: int = 50) -> torch.Tensor:
    """Lines of constant t and of constant x of the home frame, as events: (L, n, 2)."""
    lines = []
    for t0 in torch.arange(0.0, t_max + 1e-9, step, dtype=torch.float64):
        xs = torch.linspace(x_min, x_max, n, dtype=torch.float64)
        lines.append(torch.stack([torch.full_like(xs, t0), xs], dim=-1))
    for x0 in torch.arange(x_min, x_max + 1e-9, step, dtype=torch.float64):
        ts = torch.linspace(0.0, t_max, n, dtype=torch.float64)
        lines.append(torch.stack([ts, torch.full_like(ts, x0)], dim=-1))
    return torch.stack(lines)
