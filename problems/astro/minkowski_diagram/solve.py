"""Proper times, clock ticks and lines of simultaneity — then boost everything into a moving frame."""

import torch

from .equation import boost, metric, proper_time, transform


def solve(home: torch.Tensor, traveller: torch.Tensor, grid: torch.Tensor, beta_view: float) -> dict:
    """Return the world lines in the home frame S and in the frame S' moving at beta_view, plus proper times."""
    eta = metric()
    Lam = boost(torch.tensor(beta_view))  # (2, 2)
    tau_home = proper_time(home, eta)  # (N,)
    tau_trav = proper_time(traveller, eta)  # (N,)
    return {
        "S": {"home": home, "traveller": traveller, "grid": grid},
        "S'": {"home": transform(Lam, home), "traveller": transform(Lam, traveller), "grid": transform(Lam, grid)},
        "tau_home": tau_home,
        "tau_trav": tau_trav,
        "Lam": Lam,
    }


def clock_ticks(events: torch.Tensor, tau: torch.Tensor, every: float = 1.0) -> torch.Tensor:
    """Events (K, 2) where the clock on this world line shows τ = every, 2·every, … (linear interpolation)."""
    marks = torch.arange(every, float(tau[-1]) + 1e-9, every, dtype=tau.dtype)
    k = torch.searchsorted(tau, marks).clamp(1, len(tau) - 1)
    f = ((marks - tau[k - 1]) / (tau[k] - tau[k - 1]))[:, None]
    return events[k - 1] + f * (events[k] - events[k - 1])


def simultaneity_line(event: torch.Tensor, v: float, half_length: float = 12.0) -> torch.Tensor:
    """Two events (2, 2) on the line of events that an observer at `event` moving with speed v calls "now".

    In the observer's rest frame "now" is t' = const; transformed back, that is the line t - t_e = v (x - x_e).
    """
    dx = torch.tensor([-half_length, half_length], dtype=event.dtype)
    return torch.stack([event[0] + v * dx, event[1] + dx], dim=-1)
