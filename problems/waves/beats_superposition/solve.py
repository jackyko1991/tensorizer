"""Evaluate the superposition everywhere at once (no time stepping needed: every wave is known exactly)."""

import torch

from .equation import envelope, phase, superpose


def solve(A: torch.Tensor, k: torch.Tensor, omega: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> dict:
    """Return the field y (N, J), its envelope (N, J), and the time trace at x = 0 with its envelope (N,)."""
    theta = phase(k, omega, x, t)  # (S, N, J)
    y = superpose(A, theta)
    env = envelope(A, theta)
    return {"y": y, "env": env, "trace": y[:, 0], "trace_env": env[:, 0]}


def beat_frequency(trace: torch.Tensor, dt: float, f_min: float) -> float:
    """Measure the beat frequency from the time trace at one point.

    The loudness y² contains slow terms cos((ω1 − ω2) t) and fast terms near 2 f̄. Below 2 f_min only the
    slow ones are left, and the strongest of them is the beat frequency |f1 − f2|.
    """
    power = trace**2 - (trace**2).mean()
    spectrum = torch.fft.rfft(power).abs()  # (N/2 + 1,)
    freqs = torch.fft.rfftfreq(len(trace), d=dt).to(trace.device)
    slow = (freqs > 0) & (freqs < 1.9 * f_min)
    return freqs[slow][spectrum[slow].argmax()].item()
