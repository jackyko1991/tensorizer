"""Integrate the Bloch equation for all isochromats at once with RK4, switching the RF field on during pulses."""

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from .equation import bloch_rhs, omega, signal


def rf_at(t: float, pulses: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """(ω1, phase) of the RF field at time t: ω1 = flip / length inside a pulse, 0 outside."""
    for start, length, flip, phase in pulses:
        if start <= t < start + length:
            return flip / length, phase
    return 0.0, 0.0


def solve(
    M_init: torch.Tensor,
    dw: torch.Tensor,
    eps: torch.Tensor,
    R: torch.Tensor,
    M0: torch.Tensor,
    pulses: list[tuple[float, float, float, float]],
    t_max: float,
    dt: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), M (T, N, 3)).

    The field Ω_{nk} is held constant over each step (taken at the step's midpoint), so a pulse of length τ
    rotates the magnetisation by ω1·τ = the flip angle. All N isochromats advance in one batched RK4 step.
    """
    n_steps = int(round(t_max / dt))
    states = [M_init]
    M = M_init
    for n in steps(n_steps, "Bloch steps"):
        w1, phase = rf_at((n + 0.5) * dt, pulses)
        Om = omega(dw, w1, phase)  # (N, 3)
        M = rk4_step(lambda y, t, Om=Om: bloch_rhs(y, Om, eps, R, M0), M, n * dt, dt)  # Ω fixed for this step
        states.append(M)
    times = dt * torch.arange(n_steps + 1, dtype=torch.float64, device=M_init.device)
    return times, torch.stack(states)


def echo_peaks(
    times: torch.Tensor, M: torch.Tensor, TE: float, n_echoes: int, t90: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Time (K,) and height (K,) of the largest |s| within ±TE/4 of each expected echo t90 + k·TE."""
    mag = signal(M).abs()  # (T,)
    t_peak, height = [], []
    for k in range(1, n_echoes + 1):
        window = (times - (t90 + k * TE)).abs() < TE / 4
        i = torch.argmax(torch.where(window, mag, torch.zeros_like(mag)))
        t_peak.append(times[i])
        height.append(mag[i])
    return torch.stack(t_peak), torch.stack(height)


def fid_decay_time(times: torch.Tensor, M: torch.Tensor, t90: float, TE: float) -> float:
    """First time after the 90° pulse when |s| falls below 1/e: the measured T2* (before the first 180°)."""
    mag = signal(M).abs()
    after = (times > t90) & (times < t90 + TE / 2)
    below = after & (mag < torch.exp(torch.tensor(-1.0, dtype=mag.dtype)))
    if not below.any():
        return float("nan")
    return float(times[below.int().argmax()] - t90)
