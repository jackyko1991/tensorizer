"""March every ray forward in time; at each layer boundary apply Snell's law (or TIR)."""

import math

import torch

from .equation import refract
from .tensors import THICKNESS, WIDTH


def solve(
    p: torch.Tensor, d: torch.Tensor, n_layer: torch.Tensor, t_max: float, dt: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), positions (T, R, 2)).

    A ray moves at speed c/n (c = 1) along d. When a step would carry it out of its layer we stop
    it on the interface and turn its direction with refract(). Rays leaving the box freeze.
    """
    n_layers = len(n_layer)
    n_steps = int(round(t_max / dt))
    layer = torch.zeros(len(p), dtype=torch.long, device=p.device)  # (R,) layer each ray is in
    alive = torch.ones(len(p), dtype=torch.bool, device=p.device)  # (R,)
    path = [p]
    for _ in range(n_steps):
        speed = 1.0 / n_layer[layer]  # (R,)   c / n
        p_new = p + dt * speed[:, None] * d  # (R, 2)

        going_down = d[:, 1] < 0  # (R,)
        y_wall = -torch.where(going_down, layer + 1, layer).float() * THICKNESS  # (R,) boundary ahead
        next_layer = torch.where(going_down, layer + 1, layer - 1)  # (R,)  -1 or L means "outside"
        crossing = alive & torch.where(going_down, p_new[:, 1] < y_wall, p_new[:, 1] > y_wall)
        inner = crossing & (next_layer >= 0) & (next_layer < n_layers)  # a real interface, not the box edge

        if inner.any():
            # stop the ray exactly on the interface
            s = (y_wall - p[:, 1]) / (p_new[:, 1] - p[:, 1])  # (R,) fraction of the step taken
            p_hit = p + s[:, None] * (p_new - p)
            # the normal faces the incoming ray: +y for rays going down, −y for rays going up
            normal = torch.zeros_like(d)
            normal[:, 1] = torch.where(going_down, 1.0, -1.0)
            eta = n_layer[layer] / n_layer[next_layer.clamp(0, n_layers - 1)]  # (R,)   n_1 / n_2
            d_new, tir = refract(d, normal, eta)
            p_new = torch.where(inner[:, None], p_hit, p_new)
            d = torch.where(inner[:, None], d_new, d)
            layer = torch.where(inner & ~tir, next_layer, layer)  # a reflected ray stays in its layer

        p = torch.where(alive[:, None], p_new, p)  # frozen rays stay where they left the box
        alive = alive & ~(crossing & ~inner) & (p[:, 0] < WIDTH)
        path.append(p)

    times = torch.arange(n_steps + 1, device=p.device) * dt
    return times, torch.stack(path)


def critical_angle(n1: float, n2: float) -> float | None:
    """θ_c = arcsin(n2 / n1) in degrees, or None when n1 ≤ n2 (no TIR possible)."""
    return math.degrees(math.asin(n2 / n1)) if n1 > n2 else None
