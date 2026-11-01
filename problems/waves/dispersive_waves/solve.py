"""Evolve the packet exactly in k-space and measure how fast the envelope and the crests move."""

import torch

from .equation import evolve


def solve(a0: torch.Tensor, k: torch.Tensor, g: float, t_max: float, n_times: int = 181) -> tuple:
    """Return (times (N,), a (N, J) complex) — no time stepping, every time is computed directly."""
    times = torch.linspace(0.0, t_max, n_times, device=a0.device, dtype=torch.float64)
    return times, evolve(a0, k, times, g)


def centroid(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Envelope centre x̄_n = Σ_j x_j |a_nj|² / Σ_j |a_nj|².   (N, J) → (N,)"""
    w = a.abs() ** 2
    return (w @ x) / w.sum(dim=-1)


def envelope_speed(times: torch.Tensor, xbar: torch.Tensor) -> float:
    """Least-squares slope of x̄(t): the measured group velocity."""
    t = times - times.mean()
    return (torch.sum(t * (xbar - xbar.mean())) / torch.sum(t * t)).item()


def crest_speed(x: torch.Tensor, a: torch.Tensor, times: torch.Tensor, n: int = 0) -> float:
    """Measured phase velocity at the packet centre: local frequency / local wavenumber.

    ω_loc = −Δ(arg a)/Δt between two snapshots, k_loc = Δ(arg a)/Δx between two neighbouring points.
    """
    j = int(torch.argmax(a[n].abs()))
    dt, dx = (times[n + 1] - times[n]).item(), (x[1] - x[0]).item()
    w_loc = -torch.angle(a[n + 1, j] * a[n, j].conj()).item() / dt
    k_loc = torch.angle(a[n, j + 1] * a[n, j].conj()).item() / dx
    return w_loc / k_loc
