"""March the string with leap-frog and read off the reflected and transmitted amplitudes."""

import torch

from problems._common.progress import steps

from .equation import acceleration


def solve(
    u_prev: torch.Tensor, u: torch.Tensor, mu: torch.Tensor, T: float, dx: float, dt: float, t_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (S,), trajectory (S, N)). Row 0 is t = −Δt, row 1 is t = 0."""
    n_steps = int(round(t_max / dt))
    frames = [u_prev, u]
    for _ in steps(n_steps, "time steps"):
        u, u_prev = 2 * u - u_prev + dt**2 * acceleration(u, mu, T, dx), u
        frames.append(u)
    times = torch.arange(-1, n_steps + 1, device=u.device) * dt
    return times, torch.stack(frames)


def signed_peak(u: torch.Tensor) -> float:
    """The value of largest magnitude, with its sign (a pulse may be upside down)."""
    return u[u.abs().argmax()].item()


def measure(x: torch.Tensor, u: torch.Tensor) -> tuple[float, float]:
    """Reflected (x < 0) and transmitted (x > 0) pulse amplitudes in one snapshot u (N,)."""
    return signed_peak(u[x < 0]), signed_peak(u[x > 0])


def measuring_time(c1: float, c2: float, x0: float) -> float:
    """A moment when both pulses have cleared the knot but the transmitted one has not reached the far wall."""
    t_cross = abs(x0) / c1
    return t_cross + min(0.4, 0.8 / c2)
