"""Integrate the Hodgkin–Huxley equations with RK4, then count spikes and measure the firing rate."""

import torch

from problems._common.integrators import integrate

from .equation import rhs

T_ON = 10.0  # ms: the current step switches on here (the axon rests before)


def time_step(phi: float) -> float:
    """0.05 ms resolves the sub-millisecond Na gate m with RK4; warmer axons (larger φ) need proportionally less."""
    return 0.05 / max(1.0, phi)


def solve(y0: torch.Tensor, I_ext: torch.Tensor, g, E, C_m: float, phi: float, t_end: float):
    """Return (times (T,), trajectory (T, 4, B)): a current step I_b from t = T_ON to t_end, all axons at once."""
    dt = time_step(phi)
    n = int(round(t_end / dt))
    zero = torch.zeros_like(I_ext)
    traj = integrate(lambda y, t: rhs(y, I_ext if t >= T_ON else zero, g, E, C_m, phi), y0, dt, n)
    times = torch.arange(n + 1, device=y0.device) * dt
    return times, traj


def spike_times(times: torch.Tensor, V: torch.Tensor) -> list[torch.Tensor]:
    """Upward crossings of 0 mV, linearly interpolated: one tensor of spike times (ms) per axon b.   V: (T, B)"""
    up = (V[:-1] < 0) & (V[1:] >= 0)  # (T−1, B)
    out = []
    for b in range(V.shape[1]):
        i = up[:, b].nonzero().squeeze(1)
        v0, v1 = V[i, b], V[i + 1, b]
        out.append(times[i] + (0 - v0) / (v1 - v0) * (times[i + 1] - times[i]))
    return out


def firing_rate(spikes: list[torch.Tensor], t_end: float, settle: float = 40.0) -> torch.Tensor:
    """Steady firing rate in Hz from the mean interspike interval after the onset transient; 0 if it stops firing.

    Needs at least two spikes after T_ON + settle; returns a (B,) tensor.
    """
    rate = []
    for s in spikes:
        late = s[s > T_ON + settle]
        rate.append(1000.0 * (len(late) - 1) / (late[-1] - late[0]).item() if len(late) >= 2 else 0.0)
    return torch.tensor(rate)
