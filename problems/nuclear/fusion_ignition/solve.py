"""Integrate every hot spot at once with RK4 and classify ignition against quench."""

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from .equation import heating_balance, rhs

T_FLOOR = 0.5  # keV: a hot spot this cold has quenched; we freeze it


def integrate(
    y0: torch.Tensor, rho_gcc: float, loss: float, rho_r_max: float, t_max_ps: float, dt_ps: float, n_save: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times in ps (S,), states (S, B, 3)) saved at n_save evenly spaced instants."""
    n_steps = int(round(t_max_ps / dt_ps))
    save_every = max(1, n_steps // (n_save - 1))
    dt = dt_ps * 1e-12

    def f(y, _t):
        d = rhs(y, rho_gcc, loss, rho_r_max)
        frozen = (y[:, 0] < torch.log(torch.tensor(T_FLOOR, dtype=y.dtype))) | (y[:, 2] > 0.99)
        return torch.where(frozen[:, None], torch.zeros_like(d), d)

    y, times, states = y0, [0.0], [y0]
    for k in steps(n_steps, "RK4 steps"):
        y = rk4_step(f, y, k * dt, dt)
        if (k + 1) % save_every == 0:
            times.append((k + 1) * dt_ps)
            states.append(y)
    return torch.tensor(times, dtype=torch.float64), torch.stack(states)


def ignited(states: torch.Tensor) -> torch.Tensor:
    """A hot spot ignites if it at least doubles its temperature (thermal runaway): (B,) bool."""
    ln_t = states[:, :, 0]
    return ln_t.max(dim=0).values >= ln_t[0] + torch.log(torch.tensor(2.0, dtype=ln_t.dtype))


def balance_boundary(T: torch.Tensor, rho_gcc: float, loss: float, n_iter: int = 60) -> torch.Tensor:
    """ρR*(T) where self-heating just balances the losses, by bisection in ln ρR for all T at once: (n,).

    Above ρR* the hot spot heats itself (alpha absorption rises and PdV loss falls with ρR); nan if
    even ρR = 30 g/cm² cannot balance bremsstrahlung (below the ideal ignition temperature).
    """
    lo = torch.full_like(T, -8.0)  # ln ρR
    hi = torch.full_like(T, 3.4)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        heats = heating_balance(T, mid.exp(), rho_gcc, loss) > 0
        hi, lo = torch.where(heats, mid, hi), torch.where(heats, lo, mid)
    ok = heating_balance(T, torch.full_like(T, 3.4).exp(), rho_gcc, loss) > 0
    return torch.where(ok, hi.exp(), torch.full_like(T, float("nan")))


def min_ignition_rho_r(t0: torch.Tensor, r0: torch.Tensor, ign: torch.Tensor) -> torch.Tensor:
    """For each starting temperature, the smallest ρR₀ that ignites (nan if none): (n_t,)."""
    grid = ign.view(len(t0), len(r0))
    first = torch.where(grid.any(1), grid.int().argmax(1), torch.zeros(len(t0), dtype=torch.long))
    return torch.where(grid.any(1), r0[first], torch.full_like(t0, float("nan")))
