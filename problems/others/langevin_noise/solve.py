"""Euler–Maruyama time stepping of the whole ensemble: one batched step moves all B particles."""

import torch

from problems._common.progress import steps

from .equation import euler_maruyama_step


def solve(
    x0: torch.Tensor,
    v0: torch.Tensor,
    gamma: float,
    k: float,
    m: float,
    T: float,
    t_max: float,
    dt: float,
    gen: torch.Generator,
    n_saved: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (times (S,), x (S, B), v (S, B)), saved at about n_saved evenly spaced times.

    At step n we draw the row ξ_{n,:} of the noise tensor: B independent standard normals.
    """
    n_steps = int(round(t_max / dt))
    save_every = max(1, n_steps // n_saved)
    x, v = x0, v0
    xs, vs, ts = [x], [v], [0.0]
    for n in steps(n_steps, "Euler–Maruyama steps"):
        xi = torch.randn(x.shape, generator=gen, device=x.device)  # ξ_{nb} ~ N(0, 1), shape (B,)
        x, v = euler_maruyama_step(x, v, xi, gamma, k, m, T, dt)
        if (n + 1) % save_every == 0:
            xs.append(x)
            vs.append(v)
            ts.append((n + 1) * dt)
    return torch.tensor(ts, device=x.device), torch.stack(xs), torch.stack(vs)
