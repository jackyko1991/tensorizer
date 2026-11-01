"""Leapfrog time stepping of the wave field, with a line source driving a plane wave at the slits."""

import math

import torch

from .equation import CFL_LIMIT, courant, step
from .tensors import NX, NY, SOURCE_X, Setup


def solve(
    s: Setup, wavelength: float, t_max: float, dt: float = 0.5, save_every: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (times (S,), field snapshots (S, Ny, Nx), running-mean intensity on the screen (S, Ny)).

    The source adds a sin(ωt) forcing along one column (a "soft" source, so reflections pass through
    it). Wall cells are clamped to u = 0 every step. Intensity ⟨u²⟩ on the screen is averaged from
    the moment the wave front could first arrive there.
    """
    c, h = 1.0, 1.0
    if courant(c, dt, h) > CFL_LIMIT:
        raise ValueError(f"unstable: Courant number {courant(c, dt, h):.3f} > {CFL_LIMIT:.3f}")

    omega = 2 * math.pi * c / wavelength
    t_arrive = (s.screen_x - SOURCE_X) / c
    u_prev = torch.zeros(NY, NX, device=s.sigma.device)  # u^{n-1}
    u = torch.zeros_like(u_prev)  # u^n
    intensity_sum = torch.zeros(NY, device=u.device)  # Σ_n u²  on the screen column
    n_averaged = 0

    times, frames, intensity = [], [], []
    for n in range(int(round(t_max / dt))):
        t = n * dt
        u_next = step(u, u_prev, s.sigma, c, dt, h)
        ramp = min(1.0, t / (2 * wavelength))  # switch the source on smoothly over ~2 periods
        u_next[:, SOURCE_X] += dt**2 * ramp * math.sin(omega * t)
        u_next[s.wall] = 0.0  # rigid, perfectly reflecting barrier
        u_prev, u = u, u_next

        if t > t_arrive:
            intensity_sum += u[:, s.screen_x] ** 2
            n_averaged += 1
        if n % save_every == 0:
            times.append(t)
            frames.append(u.clone())
            intensity.append(intensity_sum / max(n_averaged, 1))
    return torch.tensor(times), torch.stack(frames), torch.stack(intensity)


def fringe_positions(wavelength: float, slit_sep: float, screen_dist: float, max_order: int = 3) -> list[float]:
    """Bright fringes (relative to the axis) where the path difference is mλ: y_m = L tan(arcsin(mλ/d))."""
    ys = []
    for m in range(-max_order, max_order + 1):
        sin_theta = m * wavelength / slit_sep
        if abs(sin_theta) < 1:
            ys.append(screen_dist * math.tan(math.asin(sin_theta)))
    return ys
