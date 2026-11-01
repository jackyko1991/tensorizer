"""Integrate the full mass-action kinetics and the reduced Michaelis–Menten model with RK4."""

import torch

from problems._common.integrators import integrate

from .equation import qssa_rhs, rhs


def time_step(k1, k_1, k2, S0: torch.Tensor, E0: float) -> float:
    """Binding relaxes at about k1 (S + E0) + k₋₁ + k2; keep dt below 1 / (fastest rate) for RK4."""
    k_fastest = (k1 * (S0.max() + E0) + k_1 + k2).max().item()
    return min(0.01, 1.0 / k_fastest)


def solve(y0: torch.Tensor, k1, k_1, k2, V_max, K_m, t_end: float):
    """Return (times (T,), full (T, 4, B, C), S of the reduced model (T, B, C)).

    All enzymes b and substrate concentrations c advance together in one tensor.
    """
    dt = time_step(k1, k_1, k2, y0[1], y0[0].max().item())
    n = int(round(t_end / dt))
    full = integrate(lambda y, t: rhs(y, k1, k_1, k2), y0, dt, n)  # E + S ⇌ ES → E + P
    reduced = integrate(lambda S, t: qssa_rhs(S, V_max, K_m), y0[1], dt, n)  # dS/dt = −V_max S/(K_m + S)
    times = torch.arange(n + 1, device=y0.device) * dt
    return times, full, reduced


def initial_rate(full: torch.Tensor, k2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """v0_bc = k2 · max_t ES_bc: the steady-state velocity reached right after the short pre-steady-state burst.

    Returns (S at that moment, v0), both (B, C). A little substrate is already bound or used up by then,
    so the fit uses this S rather than S0 (the difference is a few % at the lowest S0).
    """
    ES_peak, i_peak = full[:, 2].max(dim=0)  # (B, C) value and time index of the peak
    S_peak = full[:, 1].gather(0, i_peak[None]).squeeze(0)  # S_bc at that time
    return S_peak, torch.einsum("b,bc->bc", k2, ES_peak)
