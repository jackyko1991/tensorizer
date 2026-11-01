"""Spectral time stepping (every Fourier mode k at once, RK4) and the Floquet monodromy matrix, batched over k."""

from typing import Callable

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from .equation import rhs

EpsOfT = Callable[[float], float]


def evolve(u: torch.Tensor, k: torch.Tensor, eps: EpsOfT, t0: float, dt: float, n_steps: int, every: int = 1):
    """RK4 for \\dot u_{ka} = A_{kab}(t) u_{kb}. Returns the times and the states every `every` steps (incl. start)."""
    times, states = [t0], [u]
    t = t0
    for i in steps(n_steps, "RK4 steps"):
        u = rk4_step(lambda y, s: rhs(y, k, eps(s)), u, t, dt)
        t = t0 + (i + 1) * dt
        if (i + 1) % every == 0:
            times.append(t)
            states.append(u)
    return times, torch.stack(states)


def to_modes(D: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Real fields (X,) → Fourier amplitudes u_{ka} = (D_k, B_k): (K, 2) complex."""
    return torch.stack([torch.fft.fft(D), torch.fft.fft(B)], dim=-1)


def to_field(u: torch.Tensor, component: int = 0) -> torch.Tensor:
    """Fourier amplitudes (..., K, 2) → D(x) (component 0) or B(x) (component 1), shape (..., X)."""
    return torch.fft.ifft(u[..., component], dim=-1).real


def envelope(D: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Envelope |D_+(x)| of a wave packet from its analytic signal (keep only k > 0, doubled): (X,)."""
    return torch.fft.ifft(2 * torch.fft.fft(D) * (k > 0)).abs()


def interface(D0, B0, k, eps1: float, eps2: float, t_jump: float, t_max: float, dt: float, every: int):
    """A sudden jump ε1 → ε2 at t_jump, integrated in two pieces so no RK4 step straddles the jump.

    D and B (the state u) simply carry over: that *is* the continuity condition at a temporal interface.
    """
    n1 = round(t_jump / dt)
    n2 = round((t_max - t_jump) / dt)
    t_a, u_a = evolve(to_modes(D0, B0), k, lambda t: eps1, 0.0, dt, n1, every)
    t_b, u_b = evolve(u_a[-1], k, lambda t: eps2, t_a[-1], dt, n2, every)
    return torch.tensor(t_a + t_b[1:]), torch.cat([u_a, u_b[1:]])


def monodromy(k: torch.Tensor, eps: EpsOfT, period: float, n_steps: int = 400) -> torch.Tensor:
    """M_{kab}: the propagator over one period for every k, by evolving the 2×2 identity: (K, 2, 2)."""
    identity = torch.eye(2, dtype=torch.complex128, device=k.device).expand(len(k), 2, 2).clone()
    dt = period / n_steps
    M = identity
    for i in range(n_steps):
        M = rk4_step(lambda y, s: rhs(y, k, eps(s)), M, i * dt, dt)
    return M


def growth_rate_measured(spectra: torch.Tensor, times: torch.Tensor, index: int, t_fit: float) -> float:
    """Fit ln|D_k(t)| over the last part (t > t_fit) of the run for one mode: the measured growth rate."""
    keep = times > t_fit
    y = torch.log(spectra[keep, index])
    t = times[keep].to(y.dtype)
    t_c = t - t.mean()
    return (torch.sum(t_c * (y - y.mean())) / torch.sum(t_c**2)).item()
