"""Drop the ring with RK4 until its centre reaches r_stop, and compare exact and tidal-tensor deformations."""

import math

import torch

from problems._common.integrators import integrate

from .equation import rhs


def fall_time(r0: float, r: float, GM: float) -> float:
    """Newtonian time to fall from rest at r0 down to r (radial Kepler orbit, 'cycloid' solution)."""
    eta = math.acos(2 * r / r0 - 1)  # development angle: r = r0 (1 + cos η) / 2
    return math.sqrt(r0**3 / (8 * GM)) * (eta + math.sin(eta))


def solve(y0: torch.Tensor, GM: float, r_stop: float, n_steps: int) -> dict:
    """Return times (T,), the ring in absolute coordinates (T, n, 3) and its deformation, exact and linear."""
    n = (y0.shape[0] - 1) // 2
    r0 = float(y0[0, 0].norm())
    t_end = fall_time(r0, r_stop, GM)
    dt = t_end / n_steps
    traj = integrate(lambda y, t: rhs(y, GM, n), y0, dt, n_steps)  # (T, 1 + 2n, 2, 3)
    x = traj[:, :, 0]  # positions only
    centre = x[:, 0]  # (T, 3)
    return {
        "t": torch.arange(n_steps + 1, dtype=y0.dtype, device=y0.device) * dt,
        "centre": centre,
        "ring": x[:, 1 : 1 + n],  # (T, n, 3)
        "xi_exact": x[:, 1 : 1 + n] - centre[:, None],  # (T, n, 3) separation from the centre, full gravity
        "xi_tidal": x[:, 1 + n :],  # (T, n, 3) separation from the linear tidal law
    }


def stretch(xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-length of the ring along x (radial) and along y (transverse), (T,) each."""
    return xi[..., 0].abs().amax(-1), xi[..., 1].abs().amax(-1)
