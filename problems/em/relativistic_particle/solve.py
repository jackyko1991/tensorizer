"""Integrate in proper time with RK4, then resample every particle onto a common lab-time grid."""

import numpy as np
import torch

from problems._common.integrators import rk4_step

from .equation import rhs


def solve(y0: torch.Tensor, qm: torch.Tensor, F_mixed: torch.Tensor, t_max: float, dtau: float = 0.01):
    """Return (τ (T,), trajectory (T, P, 2, 4)) — all P particles advance together in proper time τ.

    We step until every particle's lab clock X^0 has passed t_max. Fast particles' clocks run ahead
    (dt/dτ = γ), so the slowest particle decides when to stop.
    """
    ys = [y0]
    while ys[-1][:, 0, 0].min() < t_max:
        ys.append(rk4_step(lambda y, tau: rhs(y, qm, F_mixed), ys[-1], 0.0, dtau))
    return torch.arange(len(ys), dtype=y0.dtype) * dtau, torch.stack(ys)


def to_lab_time(traj: torch.Tensor, t: np.ndarray) -> dict[str, np.ndarray]:
    """Resample onto lab times t (F,): positions x (F, P, 3) and γ = U^0 (F, P).

    Each particle's clock runs at its own rate (dt/dτ = γ), so equal steps in τ are unequal steps in t.
    X^0 = t grows monotonically, so we can interpolate everything else as a function of it.
    """
    X = traj[:, :, 0].numpy()  # (T, P, 4)
    U = traj[:, :, 1].numpy()  # (T, P, 4)
    n_particles = X.shape[1]
    x = np.stack([[np.interp(t, X[:, p, 0], X[:, p, i]) for i in (1, 2, 3)] for p in range(n_particles)])
    gamma = np.stack([np.interp(t, X[:, p, 0], U[:, p, 0]) for p in range(n_particles)])
    # turns of the velocity vector in the x–y plane (it starts along +y, positive charges turn clockwise)
    angle = np.unwrap(np.arctan2(U[:, :, 2], U[:, :, 1]), axis=0)  # (T, P)
    turns = np.stack([np.interp(t, X[:, p, 0], (angle[0, p] - angle[:, p]) / (2 * np.pi)) for p in range(n_particles)])
    return {"t": t, "x": x.transpose(2, 0, 1), "gamma": gamma.T, "turns": turns.T}  # (F, P, 3), (F, P), (F, P)


def newtonian(config: str, qm: torch.Tensor, y0: torch.Tensor, strength: float, t: np.ndarray) -> np.ndarray:
    """What Newton would predict, positions (F, P, 3): circles at ω = qB/m, or x = ½ a t² with a = qE/m."""
    qm, v = qm.numpy(), (y0[:, 1, 1:] / y0[:, 1, :1]).numpy()  # q/m (P,), initial 3-velocity (P, 3)
    x = np.zeros((len(t), len(qm), 3))
    if config == "cyclotron":
        omega = qm * strength  # (P,)
        radius = v[:, 1] / omega  # start moving along +y, centre at (+r, 0)
        x[:, :, 0] = radius * (1 - np.cos(np.outer(t, omega)))
        x[:, :, 1] = radius * np.sin(np.outer(t, omega))
    else:
        x[:, :, 0] = 0.5 * np.outer(t**2, qm * strength)
    return x
