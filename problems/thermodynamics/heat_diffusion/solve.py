"""Explicit (forward Euler) time stepping of ∂_t T = -∂_i q_i / (ρc)."""

import torch

from .equation import divergence_kernel, dT_dt, gradient_kernel, heat_flux, pad_boundary


def stable_dt(alpha: float, dx: float, r: float) -> float:
    """Forward Euler on the 2D Laplacian is stable only if  r = α dt / dx² ≤ 1/4."""
    if r > 0.25:
        raise ValueError(f"unstable: α dt / dx² = {r} > 1/4")
    return r * dx**2 / alpha


def solve(
    T0: torch.Tensor,
    k: float,
    rho_c: float,
    dx: float,
    t_max: float,
    r: float = 0.2,
    boundary: str = "dirichlet",
    T_edge: float = 0.0,
    n_snapshots: int = 121,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (F,), temperature snapshots (F, H, W)).

    Each step is  T ← T + dt ∂_t T : two convolutions over the whole grid at once.
    """
    dt = stable_dt(k / rho_c, dx, r)
    n_steps = max(1, int(round(t_max / dt)))
    dt = t_max / n_steps  # land exactly on t_max (only makes r slightly smaller)
    grad = gradient_kernel(dx).to(T0)  # (2, 1, 3, 3)
    div = divergence_kernel(dx).to(T0)  # (1, 2, 3, 3)
    keep = set(torch.linspace(0, n_steps, n_snapshots).round().int().tolist())  # steps to record

    T, times, frames = T0, [0.0], [T0[0, 0]]
    for step in range(1, n_steps + 1):
        T = T + dt * dT_dt(T, k, rho_c, grad, div, boundary, T_edge)
        if step in keep:
            times.append(step * dt)
            frames.append(T[0, 0])
    return torch.tensor(times), torch.stack(frames)


def flux_snapshots(frames: torch.Tensor, k: float, dx: float, boundary: str, T_edge: float) -> torch.Tensor:
    """Heat flux q_i for every snapshot, for the arrows in the plot.  (F, H, W) → (F, 2, H, W)."""
    T_padded = pad_boundary(frames[:, None], boundary, T_edge)  # (F, 1, H+4, W+4)  F acts as the batch
    q = heat_flux(T_padded, k, gradient_kernel(dx).to(frames))  # (F, 2, H+2, W+2)
    return q[:, :, 1:-1, 1:-1]  # drop the ghost faces


def heat_content(frames: torch.Tensor, rho_c: float, dx: float, T_edge: float) -> torch.Tensor:
    """Excess thermal energy per unit plate thickness  ∫ ρc (T − T_edge) dA  (J/m).  (F, H, W) → (F,)."""
    return rho_c * (frames - T_edge).sum(dim=(1, 2)) * dx**2
