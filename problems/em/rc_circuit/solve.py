"""Step all B circuits forward together with 4th-order Runge–Kutta."""

import torch

from problems._common.integrators import integrate

from .equation import current, rhs, source


def solve(V0, R, C: float, Vs0: float, shape: str, period: float, t_max: float, dt: float = 0.01) -> dict:
    """Return a dict of tensors: t (T,), Vs (T,), V (T, B), I (T, B).

    One RK4 step updates the voltage of every circuit at once: V is a tensor of shape (B,).
    """
    n_steps = int(round(t_max / dt))
    V = integrate(lambda V, t: rhs(V, source(t, Vs0, shape, period), R, C), V0, dt, n_steps)  # (T, B)
    t = torch.arange(n_steps + 1, device=V0.device) * dt  # (T,)
    Vs = torch.tensor([source(s, Vs0, shape, period) for s in t.tolist()], device=V0.device)  # (T,)
    I = current(V, Vs[:, None], R)  # (T, B)   Vs broadcast over the circuits b  # noqa: E741
    return {"t": t, "Vs": Vs, "V": V, "I": I}
