"""Integrate the ray equation with RK4 and stop each ray at the object (x = 0) or at the road (y = 0)."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(state0: torch.Tensor, dn: float, h: float, distance: float, n_steps: int = 400) -> torch.Tensor:
    """Return the ray paths (S, R, 2), S = n_steps + 1 points along the arc length s.

    The arc length runs slightly past L so every ray reaches x = 0. Once a ray is done (x ≤ 0 or y ≤ 0)
    it is frozen at its first point beyond the boundary.
    """
    ds = 1.01 * distance / n_steps
    traj = integrate(lambda y, s: rhs(y, dn, h), state0, ds, n_steps)  # (S, R, 2, 2)
    path = traj[:, :, 0, :]  # positions only

    done = (path[..., 0] <= 0) | (path[..., 1] <= 0)  # (S, R)
    done = torch.cummax(done.int(), dim=0).values.bool()  # stays True after the first hit
    first = torch.where(done.any(0), done.int().argmax(0), len(path) - 1)  # (R,) step of the hit
    end = path[first, torch.arange(path.shape[1])]  # (R, 2)
    return torch.where(done[:, :, None], end, path)


def height_at_object(path: torch.Tensor) -> torch.Tensor:
    """Height where each ray crosses x = 0 (NaN if it hit the road first): (R,).

    Interpolate linearly between the last point before x = 0 and the first one after it.
    """
    x, y = path[..., 0], path[..., 1]  # (S, R)
    heights = torch.full((path.shape[1],), float("nan"), dtype=path.dtype)
    for r in range(path.shape[1]):
        crossed = (x[:, r] <= 0).nonzero()
        if len(crossed) == 0:
            continue
        k = crossed[0, 0]
        if y[k, r] <= 0:  # reached the road on the same step: no object hit
            continue
        w = x[k - 1, r] / (x[k - 1, r] - x[k, r])  # fraction of the step until x = 0
        heights[r] = y[k - 1, r] + w * (y[k, r] - y[k - 1, r])
    return heights
