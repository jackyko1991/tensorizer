"""Integrate all alpha particles together with RK4, keeping the full path only for the ones we draw."""

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from .equation import rhs


def simulate(
    y0: torch.Tensor, k: float, m: float, dt: float, n_steps: int, n_show: int, r_detect: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (paths (T, S, 2, 2) of the first S = n_show particles, final state (B, 2, 2), detect_step (B,)).

    One RK4 step moves the whole (B, 2, 2) batch. detect_step[b] is the step at which particle b first passed
    r_detect on its way out (n_steps if never) — used to fill the angle histogram in the animation.
    """
    y = y0
    paths = [y[:n_show]]
    detect_step = torch.full((y0.shape[0],), n_steps, dtype=torch.long, device=y0.device)
    for step in steps(n_steps, "RK4 steps"):
        y = rk4_step(lambda s, _t: rhs(s, k, m), y, step * dt, dt)
        paths.append(y[:n_show])
        x, v = y[:, 0], y[:, 1]
        outgoing = (torch.einsum("bi,bi->b", x, x) > r_detect**2) & (torch.einsum("bi,bi->b", x, v) > 0)
        detect_step = torch.where(outgoing & (detect_step == n_steps), step + 1, detect_step)
    return torch.stack(paths), y, detect_step
